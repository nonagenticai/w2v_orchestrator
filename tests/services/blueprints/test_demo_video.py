"""demo_video blueprint topology + invariants."""

from __future__ import annotations

from src.services.blueprints.demo_video import DEMO_VIDEO_BLUEPRINT
from src.services.blueprints.registry import get_registry
from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS


def test_demo_video_registered() -> None:
    assert get_registry().resolve("demo_video") is DEMO_VIDEO_BLUEPRINT


def test_demo_video_topology() -> None:
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    # gen first; retention_plan then reconcile_timeline sit between gen and narrate;
    # record fans out off gen and joins reconcile_timeline; compile consumes narrate +
    # record; the L4 demo_video_footage_critique loop hangs off compile; growth_bundle
    # (off compile) and render_manifest (off the critique loop, whose post-loop artifact
    # it reads) remain the two terminal leaves. A buildable blueprint
    # already proves the DAG is acyclic (DAGBlueprint.__post_init__ raises on a cycle).
    assert order[0] == "demo_video_gen"
    assert set(order[-2:]) == {"growth_bundle", "render_manifest"}
    assert len(order) == len(DEMO_VIDEO_BLUEPRINT.stages)
    assert order.index("demo_video_gen") < order.index("retention_plan")
    assert order.index("retention_plan") < order.index("reconcile_timeline")
    assert order.index("demo_video_record") < order.index("reconcile_timeline")
    assert order.index("reconcile_timeline") < order.index("demo_video_narrate")
    assert order.index("demo_video_gen") < order.index("demo_video_narrate")
    assert order.index("demo_video_gen") < order.index("demo_video_record")
    assert order.index("demo_video_narrate") < order.index("demo_video_compile")
    assert order.index("demo_video_record") < order.index("demo_video_compile")
    assert order.index("demo_video_compile") < order.index("growth_bundle")
    assert order.index("demo_video_compile") < order.index("render_manifest")


# ── WS-2: script_doctor registered in demo_video_v1 ───────────────────────────
#
# Root cause it fixes: script_doctor was fully built + tested but had NO StageSpec
# in this blueprint (only in showcase_v1), so on every demo_video run it could not
# execute — zero rows in the live stage_outputs table. A config default alone could
# never have fixed that; the DAG has to register the stage.


def test_script_doctor_is_registered_in_demo_video() -> None:
    names = {s.name for s in DEMO_VIDEO_BLUEPRINT.stages}
    assert "script_doctor" in names
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    # A buildable blueprint already proves acyclicity (DAGBlueprint.__post_init__
    # raises "cycle in blueprint"), and a complete topological order proves every
    # stage was emitted — i.e. no stage is stranded behind a cycle.
    assert "script_doctor" in order
    assert len(order) == len(DEMO_VIDEO_BLUEPRINT.stages)
    assert len(set(order)) == len(order)


def test_script_doctor_hangs_off_gen() -> None:
    # Mirrors showcase_v1's slot exactly: the doctor reads the raw generated spine.
    assert DEMO_VIDEO_BLUEPRINT.stage("script_doctor").depends_on == ("demo_video_gen",)
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    assert order.index("demo_video_gen") < order.index("script_doctor")


def test_script_doctor_precedes_every_script_consumer() -> None:
    # The doctor rewrites line TEXT, so it must run BEFORE anything that reads the
    # final line text: claim grounding, retention rewrite/reorder, timeline
    # reconcile, voice direction, TTS narrate, footage direction, and the compile.
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    doctor = order.index("script_doctor")
    for consumer in (
        "claim_finish",
        "retention_plan",
        "reconcile_timeline",
        "voice_director",
        "demo_video_narrate",
        "demo_video_footage_direction",
        "demo_video_compile",
        "demo_video_encode_master",
    ):
        assert doctor < order.index(consumer), f"script_doctor must precede {consumer}"


def test_script_doctor_is_a_transitive_ancestor_of_narrate() -> None:
    # Ordering alone can be an accident of the topo tiebreak; assert the real EDGE
    # path so a future dep change that de-links the doctor from the render fan-out
    # fails here rather than silently making it a no-op leaf again.
    reachable = {"script_doctor"}
    for name in DEMO_VIDEO_BLUEPRINT.topological_order():
        spec = DEMO_VIDEO_BLUEPRINT.stage(name)
        if reachable.intersection(spec.depends_on):
            reachable.add(name)
    assert {"claim_finish", "demo_video_narrate", "demo_video_compile"} <= reachable
    # record reads clip ids / bdd_job_id, never line TEXT — it stays a PARALLEL
    # sibling off gen. If it were downstream of the doctor, reconcile_timeline's
    # (retention_plan, demo_video_record) join would still be acyclic, but the
    # long-pole FTR recording would serialize behind the LLM loop for nothing.
    assert "demo_video_record" not in reachable


def test_script_doctor_timeout_clears_the_llm_ladder() -> None:
    # WM evaluator-optimizer loop (<=2 passes) — pins an explicit envelope above the
    # base ceiling, same value showcase_v1 uses. test_llm_timeout_ladder enforces
    # outer > llm_budget_seconds + margin globally; this pins the demo_video value.
    from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT

    spec = DEMO_VIDEO_BLUEPRINT.stage("script_doctor")
    # 3 sequential stage_call windows x (llm_retries+1)=2 attempts x 950s + 100s slack.
    # See _LOOP_STAGE_CALLS in test_llm_timeout_ladder.py.
    assert spec.default_timeout_seconds == 5800
    assert spec.default_timeout_seconds == (
        SHOWCASE_BLUEPRINT.stage("script_doctor").default_timeout_seconds
    )


def test_script_doctor_spine_precedence_covers_demo_video() -> None:
    # The resolver both families share already ranks the doctor above the raw gen
    # spine, so registering the stage is sufficient — no consumer needed repointing.
    from src.services.stages.demo_video.spine import PLAN_STAGE_PRECEDENCE

    assert PLAN_STAGE_PRECEDENCE.index("script_doctor") < PLAN_STAGE_PRECEDENCE.index(
        "demo_video_gen"
    )
    assert PLAN_STAGE_PRECEDENCE.index("claim_finish") < PLAN_STAGE_PRECEDENCE.index(
        "script_doctor"
    )


def test_compile_depends_on_narrate_and_record() -> None:
    spec = DEMO_VIDEO_BLUEPRINT.stage("demo_video_compile")
    assert set(spec.depends_on) == {
        "demo_video_narrate",
        "demo_video_record",
        "demo_video_footage_direction",
    }


def test_narrate_off_voice_director_record_off_gen() -> None:
    # ORCH-1.3b inserted retention_plan between gen and narrate; ORCH-1.4 inserted
    # reconcile_timeline below it, then voice_director between reconcile_timeline and
    # narrate; ORCH-4.4 (A18) inserted claim_finish ABOVE retention_plan (all
    # skip-clean ⇒ byte-identical when their config blocks are absent). narrate reads
    # the (claim-softened, retention-improved, timing-reconciled) spine via
    # resolve_plan_output AND the per-line PerformanceDirective from voice_director,
    # so it hangs off voice_director; record stays wired to gen — it reads only
    # clip/id tags + expected_text, never the doctored line TEXT.
    assert DEMO_VIDEO_BLUEPRINT.stage("retention_plan").depends_on == ("claim_finish",)
    assert DEMO_VIDEO_BLUEPRINT.stage("demo_video_narrate").depends_on == ("voice_director",)
    assert DEMO_VIDEO_BLUEPRINT.stage("demo_video_record").depends_on == ("demo_video_gen",)


def test_voice_director_sits_between_reconcile_and_narrate() -> None:
    # ORCH-1.4: voice_director reads the reconciled spine and emits a per-line
    # PerformanceDirective; narrate consumes it. It does NOT rewrite line text/order,
    # so it is NOT in PLAN_STAGE_PRECEDENCE — only a new DAG edge.
    assert DEMO_VIDEO_BLUEPRINT.stage("voice_director").depends_on == ("reconcile_timeline",)
    # None => inherits the env-configurable base LLM ceiling (was a hardcoded 700).
    assert DEMO_VIDEO_BLUEPRINT.stage("voice_director").default_timeout_seconds is None
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    assert order.index("reconcile_timeline") < order.index("voice_director")
    assert order.index("voice_director") < order.index("demo_video_narrate")


def test_claim_finish_sits_above_retention_and_waits_for_the_corpus() -> None:
    # ORCH-4.4 (A18). claim_finish grounds the spine's claims against the ingested
    # source corpus, so it MUST depend on source_ingest and not merely on the plan
    # chain: source_ingest is a no-dep leaf with a 900s KAG envelope, so without the
    # explicit edge claim_finish could start while the scrape is still in flight,
    # read an empty corpus, and skip — the very inert-gate bug this PR fixes, as a
    # race. The edge never blocks: source_ingest skips clean when brief.sources is
    # absent (and claim_finish then skips too — no corpus, nothing to ground).
    # WS-2 repointed the plan-chain head onto script_doctor (registered in this
    # blueprint for the first time), so the gate now grounds the DOCTORED spine.
    spec = DEMO_VIDEO_BLUEPRINT.stage("claim_finish")
    assert set(spec.depends_on) == {"script_doctor", "source_ingest"}
    assert DEMO_VIDEO_BLUEPRINT.stage("source_ingest").depends_on == ()


def test_reconcile_timeline_joins_plan_and_footage() -> None:
    # The LAST plan mutator: it needs the doctored/retained plan AND the REAL
    # recorded footage lengths, so it depends on BOTH. record hangs off gen only —
    # a parallel sibling of the plan chain, not downstream of it — so the join is
    # cycle-free (a cycle would have raised at blueprint construction).
    assert DEMO_VIDEO_BLUEPRINT.stage("reconcile_timeline").depends_on == (
        "retention_plan",
        "demo_video_record",
    )
    # It is upstream of narrate (which consumes the per-line vo_rate it stamps) —
    # now via voice_director, the ORCH-1.4 stage inserted between the two.
    assert "reconcile_timeline" in DEMO_VIDEO_BLUEPRINT.stage("voice_director").depends_on
    order = DEMO_VIDEO_BLUEPRINT.topological_order()
    assert order.index("reconcile_timeline") < order.index("demo_video_narrate")
    # None => inherits the env-configurable base LLM ceiling. It does NOT cover this
    # stage's 4-call LLM ladder — see _LOOP_STAGE_CALLS_BACKLOG in
    # test_llm_timeout_ladder.py (KNOWN unenforced, pre-existing).
    assert DEMO_VIDEO_BLUEPRINT.stage("reconcile_timeline").default_timeout_seconds is None


def test_terminal_artifact_stage_is_compile() -> None:
    # The compile stage stays the terminal artifact (the reorder "compile cut" +
    # finalizer existence probe). The L4 footage-critique loop runs AFTER it and may
    # re-render a better video, but that accepted-best artifact reaches the
    # deliverable via render_manifest (which depends on critique), NOT by moving the
    # cut — moving it would wrongly preserve the stale stitch on a timeline reorder.
    assert DEMO_VIDEO_BLUEPRINT.terminal_artifact_stage == "demo_video_compile"
    # render_manifest hangs off the A11 ship gate, which hangs off the critique loop,
    # so it observes the pass-through best artifact (final_critique → footage_critique
    # → compile fallthrough).
    assert DEMO_VIDEO_BLUEPRINT.stage("render_manifest").depends_on == ("final_critique",)
    # The S19 encode/mux master sits between the footage-critique loop and the A11
    # ship gate; final_critique is repointed onto it (it reads the accepted-best master).
    assert DEMO_VIDEO_BLUEPRINT.stage("final_critique").depends_on == ("demo_video_encode_master",)
    assert DEMO_VIDEO_BLUEPRINT.stage("demo_video_encode_master").depends_on == (
        "demo_video_footage_critique",
    )
    assert DEMO_VIDEO_BLUEPRINT.stage("demo_video_footage_critique").depends_on == (
        "demo_video_compile",
    )
    # growth_bundle also repoints onto the ship gate (reads its badge + chapters).
    assert DEMO_VIDEO_BLUEPRINT.stage("growth_bundle").depends_on == ("final_critique",)


def test_pauses_only_after_gen_not_after_script_doctor() -> None:
    # WS-2 contract: registering script_doctor must NOT add a second human gate.
    # This family gates at demo_video_gen; showcase_v1 gates at script_doctor and
    # sets its gen-equivalent to default_pause_after=False (showcase.py:62) — so
    # the two blueprints each carry ONE gate, in different slots. Pausing here too
    # would DOUBLE-gate demo_video, not mirror showcase.
    #
    # Not a theoretical cost: the gate is inert only while script_doctor skips
    # (a SKIPPED stage never trips pause_after — dag_executor.run's
    # `and not skipped` guard), and WS-1 is about to default-inject
    # `script_doctor: {"enabled": True}`, making the stage live on every run.
    # Live prod: 68 of 91 demo_video runs are already stranded at the existing gen
    # gate vs showcase completing 22 of 25 — a second gate strands more runs and
    # defeats the point of activating the stage.
    paused = {s.name for s in DEMO_VIDEO_BLUEPRINT.stages if s.default_pause_after}
    assert paused == {"demo_video_gen"}
    assert DEMO_VIDEO_BLUEPRINT.stage("script_doctor").default_pause_after is False
    # Pin the asymmetry against showcase so a future "make them match" edit fails
    # here with the reason, rather than silently re-introducing the double gate.
    from src.services.blueprints.showcase import SHOWCASE_BLUEPRINT

    assert SHOWCASE_BLUEPRINT.stage("demo_video_gen").default_pause_after is False
    assert SHOWCASE_BLUEPRINT.stage("script_doctor").default_pause_after is True


def test_stage_timeouts_cover_inner_wait_budgets() -> None:
    # Each downstream-driving stage's envelope must exceed its inner poll budget.
    from src.services.stages.demo_video import gen as gen_mod
    from src.services.stages.demo_video import record as record_mod

    # These downstream-driving stages pin their own explicit ceiling (never inherit None).
    gen_outer = DEMO_VIDEO_BLUEPRINT.stage("demo_video_gen").default_timeout_seconds
    assert gen_outer is not None
    assert gen_outer >= gen_mod._WAIT_TIMEOUT_S
    # record re-polls the wait (1 + _WAIT_RETRIES attempts), so its envelope must
    # cover the FULL re-poll budget, not a single poll.
    record_outer = DEMO_VIDEO_BLUEPRINT.stage("demo_video_record").default_timeout_seconds
    assert record_outer is not None
    assert record_outer >= (1 + record_mod._WAIT_RETRIES) * record_mod._POLL_TIMEOUT_S


def test_poll_wait_timeouts_within_baseserviceclient_cap() -> None:
    # poll_job HARD-REJECTS any timeout > 600s (BaseServiceClient). The clients
    # are mocked in stage unit tests, so this guards the constants directly —
    # a value > 600 ValueErrors live (caught the demo-video-gen e2e regression).
    from src.services.stages.demo_video import gen as gen_mod
    from src.services.stages.demo_video import record as record_mod

    assert gen_mod._WAIT_TIMEOUT_S <= 600.0
    assert record_mod._POLL_TIMEOUT_S <= 600.0


def test_final_critique_has_an_explicit_recompile_sized_timeout() -> None:
    # REGRESSION (item 6): the A11 ship gate set NO default_timeout_seconds, so it
    # inherited the base LLM ceiling while its own inner LLM ladder needs far more, and a
    # `mode="auto"` re-drive additionally runs a FULL recompile. Tripping the outer
    # wrapper makes dag_executor raise StageResumableTimeoutError, re-dispatch from
    # scratch (new analyzer job, possibly a SECOND full compile) and — past
    # max_stage_attempts — persist `failed`: the soft gate that "never hard-blocks"
    # hard-failing the deliverable. Worse, the outer cut arrives as CancelledError (a
    # BaseException) that the stage's `except Exception` soft-degrade cannot catch.
    #
    # 2500 clears the single-stage_call invariant (950 x 2 + 30 = 1930). It does
    # NOT cover this stage's 4-call ladder (7700s) — a KNOWN, pre-existing
    # shortfall recorded in _LOOP_STAGE_CALLS_BACKLOG in
    # test_llm_timeout_ladder.py, unrelated to and untouched by the compile-wait
    # fix below.
    #
    # ⚠️ NO LONGER equal to `demo_video_compile`'s ceiling. The 2026-09-17
    # incident fix (compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS) raised
    # `demo_video_compile` far above 2500 so the orchestrator's own compile wait
    # can cover the compiler's real worst case; `final_critique` and
    # `demo_video_footage_critique` re-invoke the compile path through their OWN
    # evaluator-optimizer loops (not through the compile stage's outer ceiling),
    # and raising THEIR ceilings to match is the separately-tracked
    # _LOOP_STAGE_CALLS_BACKLOG item, not this fix's scope. Do not restore the
    # three-way equality without doing that work.
    for blueprint in (DEMO_VIDEO_BLUEPRINT, SHOWCASE_BLUEPRINT):
        assert blueprint.stage("final_critique").default_timeout_seconds == 2500
        assert blueprint.stage("demo_video_footage_critique").default_timeout_seconds == 2500
        assert (
            blueprint.stage("demo_video_compile").default_timeout_seconds
            == COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
        )
