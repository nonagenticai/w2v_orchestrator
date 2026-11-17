"""The deliverable-wiring gate: a blueprint whose terminal stage cannot supply the
handle the run finalizer will look for must fail at WIRING time, not run time.

Root cause these pin: ``DAGBlueprint.__post_init__`` validated only that
``terminal_artifact_stage`` NAMES an existing stage. Nothing checked that the
stage publishes anything the finalizer could probe, so a terminal stage that
named its deliverable something other than ``artifact_url`` left the run-level
existence gate permanently inert while its runs still reported ``completed``.
Both web-design blueprints were in that state.
"""

from __future__ import annotations

import pytest

from src.services.blueprints.base import (
    DAGBlueprint,
    DeliverableProbe,
    StageFactory,
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


class _JobOutput(StageOutput):
    """Shaped like ``AssembleOutput``: proxy links + a job id, no artifact_url."""

    redesign_job_id: str = ""
    live_html_url: str = ""
    pdf_url: str = ""


class _UrlStage(StageHandler[_Input, _UrlOutput]):
    name = "tail"
    input_model = _Input
    output_model = _UrlOutput

    async def handle(self, payload: _Input, ctx) -> _UrlOutput:
        return _UrlOutput()


class _JobStage(StageHandler[_Input, _JobOutput]):
    name = "tail"
    input_model = _Input
    output_model = _JobOutput

    async def handle(self, payload: _Input, ctx) -> _JobOutput:
        return _JobOutput()


def _bp(**kwargs) -> DAGBlueprint:
    spec = kwargs.pop("spec", StageSpec(name="tail", handler_factory=handler(_UrlStage)))
    return DAGBlueprint(
        id=kwargs.pop("id", "t_v1"),
        project_type=kwargs.pop("project_type", "t"),
        stages=[spec],
        terminal_artifact_stage="tail",
        **kwargs,
    )


def test_registering_a_blueprint_whose_terminal_stage_lacks_the_field_raises() -> None:
    """THE gate. A terminal stage with no ``artifact_url`` is rejected at register().

    Before this existed such a blueprint registered happily and every one of its
    runs completed with the deliverable unverified — silent, not loud.
    """
    bad = _bp(spec=StageSpec(name="tail", handler_factory=handler(_JobStage)))
    with pytest.raises(ValueError, match="artifact_url"):
        BlueprintRegistry().register(bad)


def test_declaring_the_real_handle_registers_cleanly() -> None:
    good = _bp(
        spec=StageSpec(name="tail", handler_factory=handler(_JobStage)),
        terminal_artifact_field="redesign_job_id",
        terminal_artifact_probe=DeliverableProbe.DESIGN_JOB,
    )
    registry = BlueprintRegistry()
    registry.register(good)
    resolved = registry.resolve("t")
    assert resolved is good
    assert resolved.terminal_artifact_field in _JobOutput.model_fields


def test_a_typoed_field_is_rejected_too() -> None:
    bad = _bp(
        spec=StageSpec(name="tail", handler_factory=handler(_JobStage)),
        terminal_artifact_field="redesign_jobid",
    )
    with pytest.raises(ValueError, match="redesign_jobid"):
        BlueprintRegistry().register(bad)


def test_an_untagged_factory_is_rejected_rather_than_skipped() -> None:
    """An un-introspectable factory is an UNANSWERED question, not a pass.

    Letting it through would restore the exact fail-open this gate exists to
    close: no answer silently reading as "fine".
    """
    bare = _bp(spec=StageSpec(name="tail", handler_factory=lambda _deps: _UrlStage()))
    with pytest.raises(ValueError, match="untagged handler_factory"):
        BlueprintRegistry().register(bare)


def test_a_hand_written_factory_wrapped_in_stage_factory_is_introspectable() -> None:
    def _make(deps: dict) -> StageHandler:
        return _UrlStage()

    spec = StageSpec(name="tail", handler_factory=StageFactory(_UrlStage, _make))
    assert stage_handler_class(spec) is _UrlStage
    # ...and it still BUILDS the stage: tagging must not replace the callable.
    assert isinstance(spec.handler_factory({}), _UrlStage)
    registry = BlueprintRegistry()
    registry.register(_bp(spec=spec))
    assert registry.resolve("t").terminal_artifact_field == "artifact_url"


def test_unknown_probe_kind_is_rejected() -> None:
    bad = _bp(terminal_artifact_probe="carrier_pigeon")
    with pytest.raises(ValueError, match="terminal_artifact_probe"):
        BlueprintRegistry().register(bad)


def test_a_blueprint_with_no_terminal_stage_is_unaffected() -> None:
    """No terminal stage ⇒ no deliverable to verify ⇒ the gate must not invent one."""
    bp = DAGBlueprint(
        id="t_v1",
        project_type="t",
        stages=[StageSpec(name="tail", handler_factory=handler(_UrlStage))],
    )
    registry = BlueprintRegistry()
    registry.register(bp)
    assert registry.resolve("t").terminal_artifact_stage is None


# --- fleet-wide ------------------------------------------------------------


def test_every_registered_blueprint_can_actually_be_verified() -> None:
    """No blueprint in the fleet ships an inert deliverable probe."""
    import src.services.blueprints  # noqa: F401  # registers all of them

    for bp in get_registry().all():
        if bp.terminal_artifact_stage is None:
            continue
        cls = stage_handler_class(bp.stage(bp.terminal_artifact_stage))
        assert cls is not None, f"{bp.id}: terminal factory is not introspectable"
        assert bp.terminal_artifact_field in cls.output_model.model_fields, (
            f"{bp.id}: {cls.output_model.__name__} declares no {bp.terminal_artifact_field!r}"
        )
        assert bp.terminal_artifact_probe in tuple(DeliverableProbe)


@pytest.mark.parametrize("project_type", ["web_design"])
def test_web_design_blueprints_probe_the_design_job(project_type: str) -> None:
    """Both were the live instances of the defect; both are now armed."""
    import src.services.blueprints  # noqa: F401

    bp = get_registry().resolve(project_type)
    assert bp.terminal_artifact_field == "redesign_job_id"
    assert bp.terminal_artifact_probe is DeliverableProbe.DESIGN_JOB
