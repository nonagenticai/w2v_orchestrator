"""Which stage holds the file the customer actually receives.

``terminal_artifact_stage`` does NOT answer that. It is the re-stitch anchor and
the finalizer's existence probe, so on the video blueprints it deliberately names
the COMPILE while the chain continues compile → footage_critique → encode_master
→ final_critique, each of which may republish a re-mastered ``artifact_url``.
Binding an assertion to the compile therefore passes green while the shipped
master is broken.

These pin the declared answer (``delivered_artifact_stage`` /
``delivered_stage()``), the DAG-derived fallback chain
(``delivered_artifact_chain()``), and — the part that matters most — that a
blueprint naming a stage that does not exist, or that publishes no deliverable
handle, fails at WIRING time rather than serving ``null`` forever.
"""

from __future__ import annotations

import pytest

from src.services.blueprints.base import (
    DAGBlueprint,
    StageSpec,
    handler,
    stage_handler_class,
)
from src.services.blueprints.registry import BlueprintRegistry, get_registry
from src.services.stages.base import StageHandler, StageInput, StageOutput


class _Input(StageInput):
    pass


class _UrlOutput(StageOutput):
    artifact_url: str = ""


class _NoUrlOutput(StageOutput):
    """Shaped like a review stage: real output, but no deliverable handle."""

    verdict: str = ""


class _Compile(StageHandler[_Input, _UrlOutput]):
    name = "compile"
    input_model = _Input
    output_model = _UrlOutput

    async def handle(self, payload: _Input, ctx) -> _UrlOutput:
        return _UrlOutput()


class _Master(StageHandler[_Input, _UrlOutput]):
    name = "master"
    input_model = _Input
    output_model = _UrlOutput

    async def handle(self, payload: _Input, ctx) -> _UrlOutput:
        return _UrlOutput()


class _Review(StageHandler[_Input, _NoUrlOutput]):
    name = "review"
    input_model = _Input
    output_model = _NoUrlOutput

    async def handle(self, payload: _Input, ctx) -> _NoUrlOutput:
        return _NoUrlOutput()


def _chain_bp(**kwargs) -> DAGBlueprint:
    """compile → master → review. Only the first two publish ``artifact_url``."""
    return DAGBlueprint(
        id=kwargs.pop("id", "t_v1"),
        project_type=kwargs.pop("project_type", "t"),
        stages=[
            StageSpec(name="compile", handler_factory=handler(_Compile)),
            StageSpec(name="master", depends_on=("compile",), handler_factory=handler(_Master)),
            StageSpec(name="review", depends_on=("master",), handler_factory=handler(_Review)),
        ],
        terminal_artifact_stage=kwargs.pop("terminal_artifact_stage", "compile"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# The declared answer
# --------------------------------------------------------------------------


def test_unset_delivered_stage_falls_back_to_the_terminal_stage() -> None:
    """A blueprint that sets nothing must still ANSWER, with the terminal stage.

    Most blueprints' compile IS their deliverable; adding the field must change
    nothing for them. If the fallback were dropped, ``delivered_stage()`` would
    return ``None`` for 14 of 16 registered blueprints and the API would report
    every audiobook/course/lesson run as having shipped nothing.
    """
    bp = _chain_bp()
    assert bp.delivered_artifact_stage is None
    assert bp.delivered_stage() == "compile"
    assert bp.delivered_stage() == bp.terminal_artifact_stage


def test_an_explicit_delivered_stage_wins_over_the_terminal_stage() -> None:
    """Explicit beats inherited — and the terminal stage is NOT disturbed by it."""
    bp = _chain_bp(delivered_artifact_stage="master")
    assert bp.delivered_stage() == "master"
    # ⛔ The re-stitch anchor and the finalizer probe must be untouched: setting
    # the delivered stage ADDS a fact, it does not repurpose this one.
    assert bp.terminal_artifact_stage == "compile"


def test_demo_video_delivers_the_pass_through_stage_not_the_compile() -> None:
    """THE case this whole field exists for, on the REAL blueprint.

    ``demo_video_v1``'s terminal stage is ``demo_video_compile`` because a
    timeline reorder must re-stitch there. Three artifact-bearing stages follow
    it. If ``delivered_stage()`` returned the compile, an E2E assertion bound to
    "the deliverable" would probe the pre-master cut and never see a broken
    re-master.
    """
    bp = get_registry().resolve_by_id("demo_video_v1")
    assert bp.terminal_artifact_stage == "demo_video_compile"
    assert bp.delivered_stage() == "final_critique"
    assert bp.delivered_stage() != bp.terminal_artifact_stage


def test_showcase_delivers_the_pass_through_stage_not_the_compile() -> None:
    """Showcase shares demo_video's tail and must give the same answer."""
    bp = get_registry().resolve_by_id("showcase_v1")
    assert bp.terminal_artifact_stage == "demo_video_compile"
    assert bp.delivered_stage() == "final_critique"


def test_promo_delivers_its_terminal_stage_not_its_compile_cut() -> None:
    """Promo is the blueprint whose terminal stage is ALREADY the deliverable.

    ``compile_cut_stage='promo_render'`` differs from
    ``terminal_artifact_stage='promo_footage_critique'``, which invites the
    mistake of "fixing" one to match the other. Nothing downstream of the
    critique publishes ``artifact_url``, so the inherited answer is the right
    one and ``promo_render`` is an INTERMEDIATE, not the delivery.
    """
    bp = get_registry().resolve_by_id("promo_v1")
    assert bp.delivered_artifact_stage is None
    assert bp.delivered_stage() == "promo_footage_critique"
    assert bp.compile_cut_stage == "promo_render"
    assert bp.delivered_stage() != bp.compile_cut_stage


# --------------------------------------------------------------------------
# The DAG-derived fallback chain
# --------------------------------------------------------------------------


def test_chain_is_most_downstream_first_and_skips_non_artifact_stages() -> None:
    """The chain must be ordered delivered-first and contain ONLY bearing stages.

    Order is the whole contract: the resolver returns the FIRST usable URL, so a
    chain ordered upstream-first would serve the pre-master compile on every run
    that re-mastered successfully — the exact inversion of the intent.
    """
    bp = _chain_bp(delivered_artifact_stage="master")
    assert bp.delivered_artifact_chain() == ("master", "compile")
    # ``review`` is downstream but publishes no ``artifact_url`` — and is also
    # not an ancestor of ``master``, so it is doubly excluded.
    assert "review" not in bp.delivered_artifact_chain()


def test_demo_video_chain_matches_final_critiques_own_fallback_order() -> None:
    """The API's chain below ``final_critique`` must BE ``_starting_artifact``'s.

    ``final_critique._starting_artifact`` reads encode_master → footage_critique
    → compile. If these two orders ever disagree, the API and the pipeline report
    different files for one run and every comparison between them is noise.
    """
    bp = get_registry().resolve_by_id("demo_video_v1")
    chain = bp.delivered_artifact_chain()
    assert chain[0] == "final_critique"
    assert chain[1:4] == (
        "demo_video_encode_master",
        "demo_video_footage_critique",
        "demo_video_compile",
    )


def test_a_blueprint_with_no_artifact_has_an_empty_chain() -> None:
    """``growth_bundle_v1`` declares no deliverable; it must say so, not guess."""
    bp = get_registry().resolve_by_id("growth_bundle_v1")
    assert bp.terminal_artifact_stage is None
    assert bp.delivered_stage() is None
    assert bp.delivered_artifact_chain() == ()


def test_every_registered_blueprint_delivers_its_most_downstream_artifact() -> None:
    """Fleet invariant, DERIVED per blueprint rather than hand-listed here.

    For every registered blueprint, the declared delivered stage must be the
    most-downstream stage reachable from the terminal one that publishes the
    deliverable field. This is what makes a future pass-through stage — someone
    appending a re-master after a compile — a RED test rather than a silently
    stale answer, on every blueprint at once.
    """
    for bp in get_registry().all():
        if bp.terminal_artifact_stage is None:
            assert bp.delivered_stage() is None, bp.id
            continue
        by_name = {s.name: s for s in bp.stages}
        # Transitive descendants of the terminal stage, computed here rather than
        # via ``delivered_artifact_chain`` so this is not a tautology.
        descendants: set[str] = set()
        stack = [bp.terminal_artifact_stage]
        while stack:
            current = stack.pop()
            for spec in bp.stages:
                if current in spec.depends_on and spec.name not in descendants:
                    descendants.add(spec.name)
                    stack.append(spec.name)
        bearing = [bp.terminal_artifact_stage]
        for name in bp.topological_order():
            if name not in descendants:
                continue
            cls = stage_handler_class(by_name[name])
            if cls is not None and bp.terminal_artifact_field in cls.output_model.model_fields:
                bearing.append(name)
        assert bp.delivered_stage() == bearing[-1], (
            f"{bp.id}: delivers {bp.delivered_stage()!r} but the most downstream "
            f"{bp.terminal_artifact_field!r}-bearing stage is {bearing[-1]!r}"
        )


# --------------------------------------------------------------------------
# Wiring failures must be LOUD, at import time
# --------------------------------------------------------------------------


def test_a_bogus_delivered_stage_name_fails_at_construction() -> None:
    """A typo must raise where the blueprint is DEFINED — i.e. at import.

    Without this the field would name nothing, ``delivered_stage()`` would return
    a name no stage answers to, and the API would serve
    ``delivered_artifact_url: null`` forever — read by every consumer as "this
    run shipped nothing" rather than "this wiring is wrong".
    """
    with pytest.raises(ValueError, match="delivered_artifact_stage 'nope' is not a stage"):
        _chain_bp(delivered_artifact_stage="nope")


def test_a_delivered_stage_publishing_no_deliverable_field_fails_at_register() -> None:
    """Existing is not enough — the stage must actually publish the handle.

    ``review`` is a real stage of this DAG, so the name check passes. Its output
    model declares no ``artifact_url``, so the URL could never resolve. That is
    worse than a typo and must fail just as loudly.
    """
    registry = BlueprintRegistry()
    bad = _chain_bp(delivered_artifact_stage="review")
    with pytest.raises(ValueError, match="declares no 'artifact_url' field"):
        registry.register(bad)


def test_a_delivered_stage_with_an_untagged_factory_fails_at_register() -> None:
    """An un-introspectable handler cannot be proven to publish the handle.

    ``handler()`` tags the factory with its class; a bare lambda does not, so
    ``stage_handler_class`` returns ``None`` and the wiring is UNVERIFIABLE.
    Accepting it would let exactly the broken shape above through unnoticed.
    """
    registry = BlueprintRegistry()
    bad = DAGBlueprint(
        id="u_v1",
        project_type="u",
        stages=[
            StageSpec(name="compile", handler_factory=handler(_Compile)),
            StageSpec(
                name="master",
                depends_on=("compile",),
                # A bare lambda, NOT ``handler()`` — that is the point: it carries no
                # class tag, so the wiring cannot be introspected.
                handler_factory=lambda _cfg: _Master(),
            ),
        ],
        terminal_artifact_stage="compile",
        delivered_artifact_stage="master",
    )
    with pytest.raises(ValueError, match="untagged handler_factory"):
        registry.register(bad)


def test_the_delivered_check_runs_even_when_there_is_no_terminal_stage() -> None:
    """The validator must not hide behind the terminal-stage early return.

    ``validate_deliverable_wiring`` returns early when
    ``terminal_artifact_stage is None``. If the delivered check sat after that
    return, a blueprint declaring only a delivered stage would skip validation
    entirely — the field's own failure mode, unguarded.
    """
    registry = BlueprintRegistry()
    bad = DAGBlueprint(
        id="n_v1",
        project_type="n",
        stages=[
            StageSpec(name="compile", handler_factory=handler(_Compile)),
            StageSpec(name="review", depends_on=("compile",), handler_factory=handler(_Review)),
        ],
        terminal_artifact_stage=None,
        delivered_artifact_stage="review",
    )
    with pytest.raises(ValueError, match="declares no 'artifact_url' field"):
        registry.register(bad)


def test_the_registry_is_populated_and_every_blueprint_names_a_real_delivered_stage() -> None:
    """Guard the POPULATION, because the sibling checks iterate it.

    Calling ``validate_deliverable_wiring()`` in a loop here would assert nothing:
    ``registry.register`` already calls it, so a blueprint that fails it never
    reaches the registry and the loop cannot fail while the import succeeds.

    What is NOT otherwise guarded is the population itself. Every delivered-artifact
    check in this file iterates ``get_registry().all()``, so an empty or truncated
    registry would turn all of them GREEN while asserting nothing about anything —
    the exact vacuity shape those checks exist to prevent. So: assert the fleet is
    actually there, and that each blueprint's delivered stage is a name the DAG
    really contains rather than a string nobody resolves.
    """
    blueprints = list(get_registry().all())
    assert len(blueprints) >= 10, (
        f"only {len(blueprints)} blueprint(s) registered — every delivered-artifact "
        "check in this file iterates this collection, so a truncated registry would "
        "make them all pass vacuously"
    )
    bearing = [bp for bp in blueprints if bp.delivered_stage() is not None]
    assert bearing, "no blueprint in the fleet delivers an artifact at all"
    for bp in bearing:
        delivered = bp.delivered_stage()
        stage_names = {stage.name for stage in bp.stages}
        assert delivered in stage_names, (
            f"{bp.id} delivers {delivered!r}, which is not a stage in its own DAG "
            f"({sorted(stage_names)})"
        )
