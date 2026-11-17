import random
from uuid import uuid4

import src.services.blueprints  # noqa: F401  (registers every blueprint)
from src.database.models import StageOutput
from src.services.blueprints.registry import get_registry
from src.services.runs.elements import (
    _ELEMENT_PATHS,
    compile_cut_stage,
    element_ids_in,
    patch_elements_in_output,
    populated_element_paths,
    reorder_elements_in_output,
    resolve_clip_spine_stage,
    stage_supports_element_pinning,
)

_CLIP_FIELDS = frozenset({"clip_lead_seconds", "clip_anchor"})


def test_script_draft_extracts_line_ids() -> None:
    out = {
        "lines": [{"id": "l1", "text": "x"}, {"id": "l2", "text": "y"}],
        "total_seconds": 10,
    }
    assert element_ids_in(out, "script_draft") == {"l1", "l2"}


def test_promo_analyze_extracts_pain_and_hook_ids() -> None:
    out = {
        "pain_points": [{"id": "p1", "text": "x"}, {"id": "p2", "text": "y"}],
        "hook_candidates": [{"id": "h1", "text": "z"}],
        "tone_hints": ["urgent"],
    }
    assert element_ids_in(out, "promo_analyze") == {"p1", "p2", "h1"}


def test_unknown_stage_returns_empty_set() -> None:
    assert element_ids_in({"anything": []}, "video_render") == set()
    assert stage_supports_element_pinning("video_render") is False


def test_app_critique_groups() -> None:
    out = {
        "pros": [{"id": "a", "text": "x", "kind": "pro"}],
        "cons": [{"id": "b", "text": "x", "kind": "con"}],
        "recommendations": [{"id": "c", "text": "x", "kind": "recommendation"}],
        "overall_take": "ok",
    }
    assert element_ids_in(out, "app_critique") == {"a", "b", "c"}


def test_lesson_paragraphs() -> None:
    out = {"paragraphs": [{"id": "p1", "text": "x", "seconds": 5.0}]}
    assert element_ids_in(out, "lesson_script") == {"p1"}


def test_app_inspect_facts() -> None:
    out = {
        "facts": [
            {"id": "f1", "kind": "title", "text": "T"},
            {"id": "f2", "kind": "feature", "text": "X"},
        ],
        "competitors": [],
    }
    assert element_ids_in(out, "app_inspect") == {"f1", "f2"}


def test_app_review_record_groups() -> None:
    out = {
        "narrations": [{"id": "n1", "line_id": "L1", "artifact_url": "u"}],
        "visuals": [{"id": "v1", "line_id": "L1"}],
    }
    assert element_ids_in(out, "app_review_record") == {"n1", "v1"}


def test_skips_nodes_without_string_id() -> None:
    out = {"lines": [{"id": 123, "text": "x"}, {"id": "good", "text": "y"}]}
    assert element_ids_in(out, "script_draft") == {"good"}


def test_supports_element_pinning_true_for_known_stages() -> None:
    for stage in ("script_draft", "promo_analyze", "lesson_script", "app_inspect"):
        assert stage_supports_element_pinning(stage) is True


def test_demo_video_gen_lines_are_pinnable() -> None:
    out = {"lines": [{"id": "clip-1", "text": "x"}, {"id": "clip-2", "text": "y"}]}
    assert stage_supports_element_pinning("demo_video_gen") is True
    assert element_ids_in(out, "demo_video_gen") == {"clip-1", "clip-2"}


# ── reorder_elements_in_output ──────────────────────────────────────────────


def test_reorder_reverses_shot_array() -> None:
    out = {
        "shots": [{"id": "s1", "seconds": 3}, {"id": "s2", "seconds": 4}, {"id": "s3"}],
        "total_seconds": 7,
    }
    new = reorder_elements_in_output(out, "storyboard_draft", ["s3", "s1", "s2"])
    assert new is not None
    assert [s["id"] for s in new["shots"]] == ["s3", "s1", "s2"]
    # Bodies untouched, siblings preserved, source not mutated.
    assert new["shots"][1]["seconds"] == 3
    assert new["total_seconds"] == 7
    assert [s["id"] for s in out["shots"]] == ["s1", "s2", "s3"]


def test_reorder_rejects_non_permutation() -> None:
    out = {"shots": [{"id": "s1"}, {"id": "s2"}]}
    assert reorder_elements_in_output(out, "storyboard_draft", ["s1"]) is None  # missing
    assert reorder_elements_in_output(out, "storyboard_draft", ["s1", "s2", "s3"]) is None  # extra
    assert reorder_elements_in_output(out, "storyboard_draft", ["s1", "s1"]) is None  # dup
    assert reorder_elements_in_output(out, "storyboard_draft", ["sX", "sY"]) is None  # unknown


def test_reorder_empty_or_unknown_stage_returns_none() -> None:
    assert reorder_elements_in_output({"shots": []}, "storyboard_draft", []) is None
    assert reorder_elements_in_output({"lines": [{"id": "a"}]}, "video_render", ["a"]) is None


# ── patch_elements_in_output ────────────────────────────────────────────────


def test_patch_merges_allowed_fields_onto_matched_clips() -> None:
    out = {
        "lines": [
            {"id": "clip-1", "text": "hi", "clip_lead_seconds": 0.0},
            {"id": "clip-2", "text": "there"},
        ],
        "total_seconds": 7,
    }
    new = patch_elements_in_output(
        out,
        "demo_video_gen",
        {"clip-1": {"clip_lead_seconds": 1.5, "clip_anchor": "start"}},
        allowed_fields=_CLIP_FIELDS,
    )
    assert new is not None
    # Matched clip gains the overrides; body + siblings untouched; source not mutated.
    assert new["lines"][0] == {
        "id": "clip-1",
        "text": "hi",
        "clip_lead_seconds": 1.5,
        "clip_anchor": "start",
    }
    assert new["lines"][1] == {"id": "clip-2", "text": "there"}
    assert new["total_seconds"] == 7
    assert out["lines"][0]["clip_lead_seconds"] == 0.0


def test_patch_skips_a_node_without_id_and_still_applies_valid_edits() -> None:
    # A malformed/degraded spine row without an "id" key must be SKIPPED, not crash
    # the whole clip-edit re-stitch with KeyError — every sibling helper guards with
    # .get("id"); the patcher must too.
    out = {
        "lines": [
            {"id": "clip-1", "text": "hi"},
            {"text": "separator row, no id"},  # id-less node
        ],
        "total_seconds": 3,
    }
    new = patch_elements_in_output(
        out,
        "demo_video_gen",
        {"clip-1": {"clip_lead_seconds": 1.5}},
        allowed_fields=_CLIP_FIELDS,
    )
    assert new is not None
    assert new["lines"][0]["clip_lead_seconds"] == 1.5  # valid edit applied
    assert new["lines"][1] == {"text": "separator row, no id"}  # id-less node untouched


def test_patch_null_clears_a_prior_override() -> None:
    out = {
        "lines": [{"id": "clip-1", "text": "hi", "clip_lead_seconds": 2.0, "clip_anchor": "end"}]
    }
    new = patch_elements_in_output(
        out, "demo_video_gen", {"clip-1": {"clip_lead_seconds": None}}, allowed_fields=_CLIP_FIELDS
    )
    assert new is not None
    assert "clip_lead_seconds" not in new["lines"][0]
    assert new["lines"][0]["clip_anchor"] == "end"  # untouched key stays


def test_patch_ignores_non_allowed_fields() -> None:
    out = {"lines": [{"id": "clip-1", "text": "hi"}]}
    new = patch_elements_in_output(
        out,
        "demo_video_gen",
        {"clip-1": {"text": "HACKED", "id": "evil", "clip_anchor": "end"}},
        allowed_fields=_CLIP_FIELDS,
    )
    assert new is not None
    # Only the whitelisted field applies; id + text are not patchable here.
    assert new["lines"][0] == {"id": "clip-1", "text": "hi", "clip_anchor": "end"}


def test_patch_partial_match_applies_live_ids_ignores_stale() -> None:
    out = {"lines": [{"id": "clip-1"}, {"id": "clip-2"}]}
    new = patch_elements_in_output(
        out,
        "demo_video_gen",
        {"clip-1": {"clip_lead_seconds": 1.0}, "clip-99": {"clip_lead_seconds": 9.0}},
        allowed_fields=_CLIP_FIELDS,
    )
    assert new is not None
    assert new["lines"][0]["clip_lead_seconds"] == 1.0
    assert "clip_lead_seconds" not in new["lines"][1]


def test_patch_returns_none_when_no_id_matches_or_stage_unknown() -> None:
    out = {"lines": [{"id": "clip-1"}]}
    # Every edited id stale → None (route → 400).
    assert (
        patch_elements_in_output(
            out,
            "demo_video_gen",
            {"clip-99": {"clip_lead_seconds": 1.0}},
            allowed_fields=_CLIP_FIELDS,
        )
        is None
    )
    # Stage carries no addressable path → None.
    assert (
        patch_elements_in_output(
            {"x": 1},
            "video_render",
            {"clip-1": {"clip_lead_seconds": 1.0}},
            allowed_fields=_CLIP_FIELDS,
        )
        is None
    )


# ── compile_cut_stage ───────────────────────────────────────────────────────


def _blueprint(stages: dict[str, tuple[str, ...]], terminal: str | None):
    from src.services.blueprints.base import DAGBlueprint, StageSpec

    specs = [
        # handler_factory is never called in these topology-only tests.
        StageSpec(name=n, handler_factory=lambda _d: None, depends_on=deps)  # ty: ignore[invalid-argument-type]
        for n, deps in stages.items()
    ]
    return DAGBlueprint(id="t", project_type="t", stages=specs, terminal_artifact_stage=terminal)


def test_compile_cut_is_terminal_stage_downstream_of_spine() -> None:
    bp = _blueprint(
        {"storyboard": (), "images": ("storyboard",), "render": ("images",)},
        terminal="render",
    )
    assert compile_cut_stage(bp, "storyboard") == "render"


def test_compile_cut_none_when_no_terminal_or_spine_not_upstream() -> None:
    no_terminal = _blueprint({"storyboard": (), "render": ("storyboard",)}, terminal=None)
    assert compile_cut_stage(no_terminal, "storyboard") is None
    bp = _blueprint({"storyboard": (), "render": ("storyboard",)}, terminal="render")
    # Spine == terminal (not strictly upstream) → nothing to re-stitch.
    assert compile_cut_stage(bp, "render") is None
    assert compile_cut_stage(bp, "missing") is None


def test_real_video_blueprints_have_upstream_spine() -> None:
    # Import triggers registration; assert the studio's spine→compile cut resolves.
    import src.services.blueprints.demo_video
    import src.services.blueprints.generic
    import src.services.blueprints.promo  # noqa: F401

    reg = get_registry()
    assert compile_cut_stage(reg.resolve("promo"), "promo_storyboard") == "promo_render"
    assert compile_cut_stage(reg.resolve("generic"), "storyboard_draft") == "video_render"
    assert compile_cut_stage(reg.resolve("demo_video"), "demo_video_gen") == "demo_video_compile"


# ── music_rewind_stage ──────────────────────────────────────────────────────
#
# This is the switch that makes a music edit rewind FAR ENOUGH on the one product
# whose cut is timed to the bed, and change nothing anywhere else.


def test_action_reel_rewinds_a_music_change_to_the_grid_owner() -> None:
    """`action_reel`'s `compile_cut_stage` is `cut_plan`, so a plain re-stitch
    already re-plans the cut — against the PRESERVED `music_select` output, i.e.
    the OLD track's beat grid. Rewinding one stage further is the entire fix."""
    from src.services.runs.elements import music_rewind_stage

    bp = get_registry().resolve_by_id("action_reel_v1")

    assert music_rewind_stage(bp) == "music_select"
    order = bp.topological_order()
    assert order.index("music_select") < order.index("cut_plan"), (
        "the rewind is only meaningful while the grid owner sits upstream of the cut"
    )


def test_every_other_blueprint_keeps_the_ordinary_restitch() -> None:
    """The regression that matters most. `sound_effects` + a compile-cut re-stitch
    is CORRECT for every deliverable type whose timeline does not depend on the
    bed, and the shared `sound_design` resolver consumes it. If this starts
    returning a stage for any of them, those runs silently begin re-running (and
    re-billing) their whole tail on a music tweak."""
    from src.services.runs.elements import music_rewind_stage

    registry = get_registry()
    rewinding = {bp.id for bp in registry.all() if music_rewind_stage(bp) is not None}

    assert rewinding == {"action_reel_v1"}, rewinding


def test_a_grid_owner_downstream_of_the_cut_is_not_a_rewind_point() -> None:
    """Re-running a stage that sits AFTER the compile cut cannot re-derive the cut,
    so the answer is None rather than a rewind that proves nothing."""
    from src.services.runs.elements import music_rewind_stage

    bp = _blueprint({"render": (), "music_select": ("render",)}, terminal="render")

    assert music_rewind_stage(bp) is None


# ── resolve_clip_spine_stage ────────────────────────────────────────────────
#
# Regression lock for live run 01a08f79-ab79-7359-9656-3238c07e19ff
# (blueprint ``demo_video_v1``): the Studio guessed the reorder spine from output
# SHAPE and 422'd, because FOUR stages of that run carry an IDENTICAL ``lines[]``
# of the same four clip ids and only ``demo_video_gen`` is a legal reorder target.
# The defect IS order-dependence, so every assertion below is made over MANY
# orderings of the same row set — a single-ordering assertion cannot fail.


def _row(stage: str, status: str = "completed", output: dict | None = None) -> StageOutput:
    """An in-memory (never persisted) stage row; only the four fields
    ``resolve_clip_spine_stage`` reads are populated."""
    return StageOutput(
        id=uuid4(),
        run_id=_RUN_ID,
        stage_name=stage,
        status=status,
        output=output,
    )


_RUN_ID = uuid4()

#: The exact clip ids the live run carried on all four look-alike stages.
_LIVE_CLIP_IDS = ["clip-1", "clip-3", "clip-4", "clip-5"]
_LIVE_LINES = {"lines": [{"id": cid, "text": f"t{cid}"} for cid in _LIVE_CLIP_IDS]}

#: The four stages of run 01a08f79 whose outputs are byte-identical in shape.
_LOOKALIKE_STAGES = ("demo_video_gen", "script_doctor", "retention_plan", "reconcile_timeline")
#: Opt-in stages that skipped clean on that run.
_LIVE_SKIPPED = ("claim_finish", "source_ingest")


def _live_demo_video_rows() -> list[StageOutput]:
    """The 16-row shape of live run 01a08f79 (all 16 demo_video_v1 stages)."""
    bp = get_registry().resolve_by_id("demo_video_v1")
    rows: list[StageOutput] = []
    for stage in bp.topological_order():
        if stage in _LIVE_SKIPPED:
            rows.append(_row(stage, status="skipped", output={}))
        elif stage in _LOOKALIKE_STAGES:
            # Identical payload on all four — this is what defeated shape-sniffing.
            rows.append(_row(stage, output=dict(_LIVE_LINES)))
        else:
            rows.append(_row(stage, output={"artifact_url": "http://a/x.mp4"}))
    assert len(rows) == 16, f"expected the live 16-stage shape, built {len(rows)}"
    return rows


def test_live_demo_video_shape_has_four_identical_line_arrays() -> None:
    """Guard the fixture itself: if the blueprint stops emitting four look-alikes
    the shuffle tests below would silently stop exercising the defect."""
    rows = _live_demo_video_rows()
    with_lines = [r.stage_name for r in rows if (r.output or {}).get("lines")]
    assert sorted(with_lines) == sorted(_LOOKALIKE_STAGES)
    payloads = [r.output for r in rows if (r.output or {}).get("lines")]
    assert all(p == payloads[0] for p in payloads), "look-alike outputs must be identical"


def test_resolve_clip_spine_is_independent_of_row_order_demo_video() -> None:
    """demo_video_v1: ``demo_video_gen`` wins for EVERY ordering of the 16 rows."""
    bp = get_registry().resolve_by_id("demo_video_v1")
    rows = _live_demo_video_rows()
    rng = random.Random(1337)  # noqa: S311 - shuffling test rows, not keys
    for _ in range(200):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert resolve_clip_spine_stage(bp, shuffled) == "demo_video_gen", (
            f"order-dependent result for {[r.stage_name for r in shuffled]}"
        )
    # Explicit extremes: the spine FIRST and the spine LAST.
    gen = next(r for r in rows if r.stage_name == "demo_video_gen")
    others = [r for r in rows if r.stage_name != "demo_video_gen"]
    assert resolve_clip_spine_stage(bp, [gen, *others]) == "demo_video_gen"
    assert resolve_clip_spine_stage(bp, [*others, gen]) == "demo_video_gen"


def _generic_two_spine_rows() -> list[StageOutput]:
    """generic_v1 rows where TWO stages qualify as a spine.

    ``demo_video_v1`` has exactly one pinnable stage, so it cannot distinguish
    "walk the blueprint in reverse topological order" from "take the first
    qualifying ROW". generic_v1 can: ``script_draft`` and ``storyboard_draft``
    both carry element arrays and both sit strictly upstream of ``video_render``,
    and the contract is that the LATER one topologically wins regardless of where
    either row appears in the list.
    """
    bp = get_registry().resolve_by_id("generic_v1")
    rows: list[StageOutput] = []
    for stage in bp.topological_order():
        if stage == "script_draft":
            rows.append(_row(stage, output={"lines": [{"id": "L1"}, {"id": "L2"}]}))
        elif stage == "storyboard_draft":
            rows.append(_row(stage, output={"shots": [{"id": "s1"}, {"id": "s2"}]}))
        else:
            rows.append(_row(stage, output={"artifact_url": "http://a/x.mp4"}))
    return rows


def test_resolve_clip_spine_prefers_latest_topological_stage_not_first_row() -> None:
    """Two qualifying stages: the topologically LATER one wins from any ordering."""
    bp = get_registry().resolve_by_id("generic_v1")
    rows = _generic_two_spine_rows()
    script = next(r for r in rows if r.stage_name == "script_draft")
    board = next(r for r in rows if r.stage_name == "storyboard_draft")
    rest = [r for r in rows if r.stage_name not in ("script_draft", "storyboard_draft")]

    # The killer ordering: the WRONG stage is the first (and the last) row.
    assert resolve_clip_spine_stage(bp, [script, board, *rest]) == "storyboard_draft"
    assert resolve_clip_spine_stage(bp, [board, script, *rest]) == "storyboard_draft"
    assert resolve_clip_spine_stage(bp, [*rest, script, board]) == "storyboard_draft"
    assert resolve_clip_spine_stage(bp, [*rest, board, script]) == "storyboard_draft"

    rng = random.Random(24601)  # noqa: S311 - shuffling test rows, not keys
    for _ in range(200):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert resolve_clip_spine_stage(bp, shuffled) == "storyboard_draft", (
            f"order-dependent result for {[r.stage_name for r in shuffled]}"
        )


def test_resolve_clip_spine_none_when_nothing_qualifies() -> None:
    bp = get_registry().resolve_by_id("demo_video_v1")
    rows = _live_demo_video_rows()
    by_name = {r.stage_name: r for r in rows}

    # (1) The spine never completed — every look-alike is still there and
    #     completed, so a shape-sniffing implementation would still answer.
    by_name["demo_video_gen"].status = "running"
    assert resolve_clip_spine_stage(bp, rows) is None
    rng = random.Random(99)  # noqa: S311 - shuffling test rows, not keys
    for _ in range(50):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert resolve_clip_spine_stage(bp, shuffled) is None

    # (2) Completed but with NO output at all.
    by_name["demo_video_gen"].status = "completed"
    by_name["demo_video_gen"].output = None
    assert resolve_clip_spine_stage(bp, rows) is None

    # (3) Completed with an output that carries an EMPTY spine — not orderable.
    by_name["demo_video_gen"].output = {"lines": []}
    assert resolve_clip_spine_stage(bp, rows) is None

    # (4) No rows at all.
    assert resolve_clip_spine_stage(bp, []) is None


def test_resolve_clip_spine_result_is_always_accepted_by_the_reorder_guard() -> None:
    """Whatever it returns must survive BOTH gates the reorder route applies —
    otherwise the endpoint 422s on the backend's own answer."""
    for bp_id, rows in (
        ("demo_video_v1", _live_demo_video_rows()),
        ("generic_v1", _generic_two_spine_rows()),
    ):
        bp = get_registry().resolve_by_id(bp_id)
        spine = resolve_clip_spine_stage(bp, rows)
        assert spine is not None
        assert stage_supports_element_pinning(spine) is True
        assert compile_cut_stage(bp, spine) is not None


def _app_review_rows(*, record_visuals: bool) -> list[StageOutput]:
    """app_review_v1 rows where the LAST pinnable stage is a MULTI-array stage.

    ``app_review_record`` declares ``["narrations", "visuals"]`` and on a real run
    populates both. It is not a timeline: ``element_ids_in`` returns the UNION of
    the two arrays' ids, and ``reorder_elements_in_output`` demands ``ordered_ids``
    be a permutation of exactly that union — i.e. the client would have to
    interleave narration ids and visual ids into one flat list, which orders
    nothing. The spine must fall through to ``app_review_script`` (one ``lines``
    array). ``record_visuals=False`` builds the same stage with only ONE array
    populated, which IS a legal spine.
    """
    bp = get_registry().resolve_by_id("app_review_v1")
    rows: list[StageOutput] = []
    for stage in bp.topological_order():
        if stage == "app_review_script":
            rows.append(_row(stage, output={"lines": [{"id": "L1"}, {"id": "L2"}]}))
        elif stage == "app_review_record":
            out: dict = {"narrations": [{"id": "n1"}, {"id": "n2"}]}
            if record_visuals:
                out["visuals"] = [{"id": "v1"}, {"id": "v2"}]
            rows.append(_row(stage, output=out))
        else:
            rows.append(_row(stage, output={"artifact_url": "http://a/x.mp4"}))
    return rows


def test_resolve_clip_spine_skips_a_multi_array_stage_for_the_upstream_timeline() -> None:
    """app_review_v1: ``app_review_record`` carries BOTH narrations and visuals,
    so it is skipped and ``app_review_script`` — the single-array stage — wins."""
    bp = get_registry().resolve_by_id("app_review_v1")
    rows = _app_review_rows(record_visuals=True)

    assert resolve_clip_spine_stage(bp, rows) == "app_review_script"

    # The defect is topological, not positional: assert over many orderings.
    rng = random.Random(5150)  # noqa: S311 - shuffling test rows, not keys
    for _ in range(200):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert resolve_clip_spine_stage(bp, shuffled) == "app_review_script", (
            f"order-dependent result for {[r.stage_name for r in shuffled]}"
        )

    # And whatever it picks must survive the reorder route's own gates.
    assert (
        reorder_elements_in_output(
            next(r for r in rows if r.stage_name == "app_review_script").output or {},
            "app_review_script",
            ["L2", "L1"],
        )
        is not None
    )


def test_resolve_clip_spine_accepts_a_multi_path_stage_that_populated_only_one() -> None:
    """A stage DECLARED with several paths but carrying ONE populated array on
    this run is a genuine single-array spine and must still qualify.

    Guards against implementing the rule as ``len(_ELEMENT_PATHS[stage]) > 1``,
    which would reject ``app_review_record`` even on a run with no visuals.
    """
    bp = get_registry().resolve_by_id("app_review_v1")
    rows = _app_review_rows(record_visuals=False)

    assert len(_ELEMENT_PATHS["app_review_record"]) > 1, "fixture needs a multi-path stage"
    record = next(r for r in rows if r.stage_name == "app_review_record")
    assert populated_element_paths(record.output or {}, "app_review_record") == ["narrations"]

    assert resolve_clip_spine_stage(bp, rows) == "app_review_record"

    rng = random.Random(90210)  # noqa: S311 - shuffling test rows, not keys
    for _ in range(200):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert resolve_clip_spine_stage(bp, shuffled) == "app_review_record", (
            f"order-dependent result for {[r.stage_name for r in shuffled]}"
        )
