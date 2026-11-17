import pytest

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.blueprints.registry import BlueprintRegistry
from src.services.stages.base import StageHandler, StageInput, StageOutput


class _NoopIn(StageInput):
    pass


class _NoopOut(StageOutput):
    pass


class _NoopHandler(StageHandler[_NoopIn, _NoopOut]):
    name = "noop"
    input_model = _NoopIn
    output_model = _NoopOut

    async def handle(self, payload: _NoopIn, ctx: object) -> _NoopOut:
        return _NoopOut()


def _noop_factory(deps: dict) -> _NoopHandler:
    return _NoopHandler()


def test_topological_order() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=_noop_factory),
            StageSpec(name="b", depends_on=("a",), handler_factory=_noop_factory),
            StageSpec(name="c", depends_on=("a",), handler_factory=_noop_factory),
            StageSpec(name="d", depends_on=("b", "c"), handler_factory=_noop_factory),
        ],
    )
    order = bp.topological_order()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_descendants() -> None:
    # gen ─▶ mid ─▶ tail ; ingest ─▶ tail (ingest is a parallel root of gen).
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="gen", handler_factory=_noop_factory),
            StageSpec(name="ingest", handler_factory=_noop_factory),
            StageSpec(name="mid", depends_on=("gen",), handler_factory=_noop_factory),
            StageSpec(name="tail", depends_on=("mid", "ingest"), handler_factory=_noop_factory),
        ],
    )
    # Transitive dependents of gen, excluding gen itself; ingest is NOT reachable.
    assert bp.descendants("gen") == {"mid", "tail"}
    assert "ingest" not in bp.descendants("gen")
    # A parallel root reaches only the join and beyond.
    assert bp.descendants("ingest") == {"tail"}
    # A leaf has no descendants.
    assert bp.descendants("tail") == set()


def test_descendants_unknown_stage_raises() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="a", handler_factory=_noop_factory)],
    )
    with pytest.raises(KeyError):
        bp.descendants("nope")


def test_cycle_detection() -> None:
    with pytest.raises(ValueError, match="cycle"):
        DAGBlueprint(
            id="t",
            project_type="generic",
            stages=[
                StageSpec(name="a", depends_on=("b",), handler_factory=_noop_factory),
                StageSpec(name="b", depends_on=("a",), handler_factory=_noop_factory),
            ],
        )


def test_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown dependency"):
        DAGBlueprint(
            id="t",
            project_type="generic",
            stages=[
                StageSpec(name="a", depends_on=("nope",), handler_factory=_noop_factory),
            ],
        )


def test_parallel_groups() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=_noop_factory),
            StageSpec(
                name="b",
                depends_on=("a",),
                parallel_group="g1",
                handler_factory=_noop_factory,
            ),
            StageSpec(
                name="c",
                depends_on=("a",),
                parallel_group="g1",
                handler_factory=_noop_factory,
            ),
        ],
    )
    groups = bp.parallel_groups()
    assert "g1" in groups
    assert set(groups["g1"]) == {"b", "c"}


def test_registry_register_and_resolve() -> None:
    reg = BlueprintRegistry()
    bp = DAGBlueprint(
        id="g",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=_noop_factory),
        ],
    )
    reg.register(bp)
    assert reg.resolve("generic") is bp
    with pytest.raises(KeyError):
        reg.resolve("course")
