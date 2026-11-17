"""ORCH-1.2 script_doctor — showcase DAG wiring + spine-resolver repoint.

Structure + pure resolver only (no WM, no run engine): the blueprint topology
proves the doctor stage was inserted and the render fan-out repointed onto it,
and the pure spine resolver proves the DOCTORED text wins when the doctor ran and
falls back to demo_video_gen when it skipped (the split-brain the repoint hangs on).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.services.blueprints.registry import get_registry
from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.spine import resolve_plan_lines, resolve_plan_output


def _ctx(prior: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {}},
        prior_run_outputs=prior,
    )


# Same shape (ids/order preserved); only the line TEXT differs between them.
_GEN: dict[str, Any] = {"lines": [{"id": "clip-1", "text": "raw gen line"}], "bdd_job_id": "b1"}
_DOCTORED: dict[str, Any] = {
    "lines": [{"id": "clip-1", "text": "doctored line"}],
    "bdd_job_id": "b1",
}
_RETAINED: dict[str, Any] = {
    "lines": [{"id": "clip-1", "text": "retained line"}],
    "bdd_job_id": "b1",
}
# ORCH-1.4 reconcile_timeline re-emits the same shape and stamps per-line timing.
_RECONCILED: dict[str, Any] = {
    "lines": [{"id": "clip-1", "text": "retained line", "vo_rate": 1.1}],
    "bdd_job_id": "b1",
}


# ── blueprint topology ────────────────────────────────────────────────────


def test_showcase_builds_with_script_doctor() -> None:
    # DAGBlueprint.__post_init__ validates every dep exists and rejects cycles, so a
    # registered, buildable blueprint proves script_doctor's dep resolves and the
    # fan-out repoint introduced no cycle.
    assert get_registry().resolve("showcase") is SHOWCASE_BLUEPRINT
    assert "script_doctor" in {s.name for s in SHOWCASE_BLUEPRINT.stages}
    order = SHOWCASE_BLUEPRINT.topological_order()
    assert order.index("demo_video_gen") < order.index("script_doctor")


def test_script_doctor_sits_between_gen_and_fanout() -> None:
    # script_doctor still hangs off gen; ORCH-1.3b inserted retention_plan between
    # script_doctor and the fan-out, and ORCH-1.4 inserted reconcile_timeline below
    # retention_plan, so the render fan-out is now repointed onto reconcile_timeline —
    # script_doctor stays a transitive ancestor of every consumer, and the
    # latest-spine stage the fan-out reads via resolve_plan_output is
    # reconcile_timeline. ORCH-4.4 (A18) then inserted claim_finish between
    # script_doctor and retention_plan: it softens the already-doctored lines, and
    # its word-count changes are seen by retention's reorder and A22's reconcile.
    assert SHOWCASE_BLUEPRINT.stage("script_doctor").depends_on == ("demo_video_gen",)
    assert set(SHOWCASE_BLUEPRINT.stage("claim_finish").depends_on) == {
        "script_doctor",
        "source_ingest",
    }
    assert SHOWCASE_BLUEPRINT.stage("retention_plan").depends_on == ("claim_finish",)
    assert SHOWCASE_BLUEPRINT.stage("reconcile_timeline").depends_on == (
        "retention_plan",
        "demo_video_record",
    )
    # ORCH-1.4 inserted voice_director between reconcile_timeline and the narrate
    # fan-out: narrate now reads voice_director (a transitive child of
    # reconcile_timeline), while the other fan-out consumers still read
    # reconcile_timeline directly.
    assert SHOWCASE_BLUEPRINT.stage("voice_director").depends_on == ("reconcile_timeline",)
    assert SHOWCASE_BLUEPRINT.stage("demo_video_narrate").depends_on == ("voice_director",)
    for stage in ("showcase_slides", "demo_video_sound_director"):
        assert "reconcile_timeline" in SHOWCASE_BLUEPRINT.stage(stage).depends_on
    order = SHOWCASE_BLUEPRINT.topological_order()
    assert order.index("script_doctor") < order.index("retention_plan")
    assert order.index("retention_plan") < order.index("reconcile_timeline")
    assert order.index("reconcile_timeline") < order.index("voice_director")
    assert order.index("voice_director") < order.index("demo_video_narrate")


def test_record_still_reads_raw_gen() -> None:
    # record needs bdd_job_id / package_url passthrough, NOT doctored text, so it
    # stays wired to demo_video_gen (not the doctor).
    assert SHOWCASE_BLUEPRINT.stage("demo_video_record").depends_on == ("demo_video_gen",)


def test_terminal_artifact_stage_unchanged() -> None:
    # The L4 footage-critique loop does NOT move the terminal artifact: compile stays
    # the reorder cut + finalizer probe; the loop's re-render reaches the deliverable
    # via render_manifest (which depends on critique).
    assert SHOWCASE_BLUEPRINT.terminal_artifact_stage == "demo_video_compile"


def test_script_doctor_pauses_for_review() -> None:
    assert SHOWCASE_BLUEPRINT.stage("script_doctor").default_pause_after is True


# ── spine resolver: split-brain (active vs skipped doctor) ─────────────────


def test_resolver_prefers_doctored_when_doctor_ran() -> None:
    ctx = _ctx({"demo_video_gen": _GEN, "script_doctor": _DOCTORED})
    assert resolve_plan_output(ctx) == _DOCTORED
    assert resolve_plan_lines(ctx) == _DOCTORED["lines"]
    assert resolve_plan_lines(ctx)[0]["text"] == "doctored line"


def test_resolver_falls_back_to_gen_when_doctor_absent() -> None:
    ctx = _ctx({"demo_video_gen": _GEN})
    assert resolve_plan_output(ctx) == _GEN
    assert resolve_plan_lines(ctx) == _GEN["lines"]
    assert resolve_plan_lines(ctx)[0]["text"] == "raw gen line"


def test_resolver_empty_when_neither_present() -> None:
    ctx = _ctx({})
    assert resolve_plan_output(ctx) == {}
    assert resolve_plan_lines(ctx) == []


def test_resolver_prefers_reconciled_over_retention_plan() -> None:
    # reconcile_timeline is the LAST plan mutator, so its output wins the whole chain.
    ctx = _ctx(
        {
            "demo_video_gen": _GEN,
            "script_doctor": _DOCTORED,
            "retention_plan": _RETAINED,
            "reconcile_timeline": _RECONCILED,
        }
    )
    assert resolve_plan_output(ctx) == _RECONCILED
    assert resolve_plan_lines(ctx)[0]["vo_rate"] == 1.1


def test_resolver_falls_through_when_reconcile_skipped() -> None:
    # A skipped stage stores a falsy `{}`, so the `or` chain falls through to
    # retention_plan UNCHANGED — the byte-identity guarantee for a run with no
    # `direction.timing` block.
    skipped = _ctx(
        {
            "demo_video_gen": _GEN,
            "script_doctor": _DOCTORED,
            "retention_plan": _RETAINED,
            "reconcile_timeline": {},
        }
    )
    absent = _ctx({"demo_video_gen": _GEN, "script_doctor": _DOCTORED, "retention_plan": _RETAINED})
    assert resolve_plan_output(skipped) == _RETAINED
    assert resolve_plan_output(absent) == _RETAINED
    assert resolve_plan_output(skipped) == resolve_plan_output(absent)
    assert resolve_plan_lines(skipped)[0]["text"] == "retained line"


# ── inert: skip-clean doctor ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_script_doctor_skips_clean_when_no_block() -> None:
    # No enabled `script_doctor` config block ⇒ StageSkippedError (skip-clean). A
    # skipped stage never pauses regardless of default_pause_after, and the resolver
    # falls back to demo_video_gen — so an inert doctor is byte-identical to the
    # pre-ORCH-1.2 topology. Exercises the parallel-built ScriptDoctorStage.
    from src.services.stages.shared.script_doctor import ScriptDoctorStage

    stage = ScriptDoctorStage()
    ctx = _ctx({"demo_video_gen": _GEN})
    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)
