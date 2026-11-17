"""showcase blueprint topology + invariants."""

from __future__ import annotations

from src.services.blueprints.registry import get_registry
from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT


def test_showcase_registered() -> None:
    assert get_registry().resolve("showcase") is SHOWCASE_BLUEPRINT
    assert SHOWCASE_BLUEPRINT.id == "showcase_v1"


def test_showcase_topology() -> None:
    order = SHOWCASE_BLUEPRINT.topological_order()
    # gen first; narrate + record + slides all fan out off gen; compile joins all
    # three; the L4 demo_video_footage_critique loop hangs off compile; growth_bundle
    # (off compile) and render_manifest (off the critique loop) remain the two leaves.
    assert order[0] == "demo_video_gen"
    assert set(order[-2:]) == {"growth_bundle", "render_manifest"}
    for stage in ("demo_video_narrate", "demo_video_record", "showcase_slides"):
        assert order.index("demo_video_gen") < order.index(stage)
        assert order.index(stage) < order.index("demo_video_compile")
    assert order.index("demo_video_compile") < order.index("growth_bundle")
    assert order.index("demo_video_compile") < order.index("render_manifest")


def test_compile_depends_on_narrate_record_slides_and_sound() -> None:
    spec = SHOWCASE_BLUEPRINT.stage("demo_video_compile")
    assert set(spec.depends_on) == {
        "demo_video_narrate",
        "demo_video_record",
        "showcase_slides",
        "demo_video_sound_director",
        "demo_video_footage_direction",
    }


def test_sound_director_reads_gen_and_narrate() -> None:
    # Needs narration TEXT (the reconcile_timeline spine, which falls back to
    # retention_plan then script_doctor then gen) + real per-line DURATIONS (narrate)
    # to place accents; opt-in skip-clean keeps it from blocking a run without
    # sound_effects. ORCH-1.4 repointed the retention_plan edge onto
    # reconcile_timeline (the latest-spine stage).
    spec = SHOWCASE_BLUEPRINT.stage("demo_video_sound_director")
    assert set(spec.depends_on) == {"reconcile_timeline", "demo_video_narrate"}


def test_slides_depends_on_gen_and_narrate() -> None:
    # narrate too: so each card's motion clip is rendered for its full narration
    # length (animated the whole time on screen, no last-frame freeze). The
    # spine edge is reconcile_timeline (ORCH-1.4) — the latest-spine producer.
    assert set(SHOWCASE_BLUEPRINT.stage("showcase_slides").depends_on) == {
        "reconcile_timeline",
        "demo_video_narrate",
    }


def test_reconcile_timeline_joins_plan_and_footage() -> None:
    # The LAST plan mutator: it needs the doctored/retained plan AND the REAL
    # recorded footage lengths. record hangs off gen only — a parallel sibling of the
    # plan chain — so the join is cycle-free (a cycle would raise at construction).
    assert SHOWCASE_BLUEPRINT.stage("reconcile_timeline").depends_on == (
        "retention_plan",
        "demo_video_record",
    )
    # None => inherits the env-configurable base LLM ceiling. It does NOT cover this
    # stage's 4-call LLM ladder — see _LOOP_STAGE_CALLS_BACKLOG in
    # test_llm_timeout_ladder.py (KNOWN unenforced, pre-existing).
    assert SHOWCASE_BLUEPRINT.stage("reconcile_timeline").default_timeout_seconds is None
    order = SHOWCASE_BLUEPRINT.topological_order()
    assert len(order) == len(SHOWCASE_BLUEPRINT.stages)
    assert order.index("retention_plan") < order.index("reconcile_timeline")
    assert order.index("demo_video_record") < order.index("reconcile_timeline")
    # Upstream of the whole render fan-out (each consumes the vo_rate it stamps).
    for stage in ("demo_video_narrate", "showcase_slides", "demo_video_sound_director"):
        assert order.index("reconcile_timeline") < order.index(stage)


def test_voice_director_sits_between_reconcile_and_narrate() -> None:
    # ORCH-1.4: voice_director reads the reconciled spine and emits a per-line
    # PerformanceDirective; narrate consumes it via ctx.prior_output. It does NOT
    # rewrite line text/order (not in PLAN_STAGE_PRECEDENCE) — only a new DAG edge.
    assert SHOWCASE_BLUEPRINT.stage("voice_director").depends_on == ("reconcile_timeline",)
    # None => inherits the env-configurable base LLM ceiling (was a hardcoded 700).
    assert SHOWCASE_BLUEPRINT.stage("voice_director").default_timeout_seconds is None
    assert SHOWCASE_BLUEPRINT.stage("demo_video_narrate").depends_on == ("voice_director",)
    order = SHOWCASE_BLUEPRINT.topological_order()
    assert order.index("reconcile_timeline") < order.index("voice_director")
    assert order.index("voice_director") < order.index("demo_video_narrate")


def test_terminal_artifact_stage_is_compile() -> None:
    # The compile stage stays the terminal artifact (reorder "compile cut" + finalizer
    # existence probe). The L4 footage-critique loop runs after it; its accepted-best
    # re-render reaches the deliverable via render_manifest (which depends on
    # critique), NOT by moving the cut.
    assert SHOWCASE_BLUEPRINT.terminal_artifact_stage == "demo_video_compile"
    # render_manifest hangs off the A11 ship gate, which hangs off the critique loop.
    assert SHOWCASE_BLUEPRINT.stage("render_manifest").depends_on == ("final_critique",)
    # The S19 encode/mux master sits between the footage-critique loop and the A11 ship
    # gate; final_critique is repointed onto it (it reads the accepted-best master).
    assert SHOWCASE_BLUEPRINT.stage("final_critique").depends_on == ("demo_video_encode_master",)
    assert SHOWCASE_BLUEPRINT.stage("demo_video_encode_master").depends_on == (
        "demo_video_footage_critique",
    )
    assert SHOWCASE_BLUEPRINT.stage("demo_video_footage_critique").depends_on == (
        "demo_video_compile",
    )
    assert SHOWCASE_BLUEPRINT.stage("growth_bundle").depends_on == ("final_critique",)


def test_authored_gen_does_not_pause() -> None:
    # Authored input — nothing GENERATED to review, so demo_video_gen runs straight
    # through (unlike demo_video, which pauses after its generated gen). The only
    # pause beat is the ORCH-1.2 script_doctor, whose DOCTORED script is worth a
    # review before the render fan-out (and which is skipped entirely when inert).
    paused = {s.name for s in SHOWCASE_BLUEPRINT.stages if s.default_pause_after}
    assert paused == {"script_doctor"}
    assert not SHOWCASE_BLUEPRINT.stage("demo_video_gen").default_pause_after
