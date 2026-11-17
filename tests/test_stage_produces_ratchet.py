"""CI-enforced ratchet for the ``produces`` stage-output contract.

A stage that catches its own failure, writes an empty artifact and reports
``completed`` is the most expensive recurring defect class in this service: the
run is green, the deliverable is unusable, and nothing downstream disagrees.
It has been re-implemented independently in a dozen stage modules, each locally
reasonable, so fixing them one at a time does not converge.

``StageOutput.produces`` is the contract that closes it, enforced at the single
chokepoint every stage's success passes through
(``runs/dag_executor.py::_enforce_produces``). This file is what makes it spread:
the number of stage outputs NOT declaring ``produces`` is a budget that may only
ever go DOWN. A new stage added without a declaration fails CI.

The second test is the one that keeps the mechanism honest — a declaration is
useless if the stage has no ``notes`` channel to degrade through, so the two are
required together.
"""

from __future__ import annotations

import importlib
import pkgutil

from src.services.stages.base import StageHandler, StageOutput, StageOutputWithNotes

# May only ever be LOWERED. Raising it means a new stage shipped without saying
# what it produces — declare `produces` on it instead of moving this number.
#
# Raised 38 -> 39 for `WebsiteReviewSlidePlanOutput`: this is the one legitimate
# exception the rule above does not cover. `produces` with the (default)
# non-required tier DEGRADES an empty declared field, and an empty plan is a
# CORRECT, common outcome here — the operator authored the deck by hand
# (`source="authored_shotlist"`), the run has no BEFORE-act lines
# (`source="no_lines"`), or it is a critique-only run. Declaring `produces` would
# mark every one of those healthy runs `degraded`. Its sibling
# `WebsiteReviewSlidesOutput` (already in the undeclared set) documents the
# identical reasoning for the same reason — see its docstring.
MAX_UNDECLARED_STAGE_OUTPUTS = 39

# The other direction of the same ratchet, and the one that actually has teeth.
# `produces` alone only DEGRADES: a violation appends a note and the run is
# surfaced as `degraded`, never failed. Until W1-1 every declaring stage was
# degrade-only, so the whole contract could not fail anything — a compile stage
# that produced no artifact still reached `completed`. `produces_required` is the
# half that fails, and this floor is what stops it being quietly reverted to zero
# by a refactor: it may only ever be RAISED.
MIN_REQUIRED_STAGE_OUTPUTS = 15


def _import_every_stage_module() -> None:
    """Stage classes only exist once their module is imported."""
    package = importlib.import_module("src.services.stages")
    for info in pkgutil.walk_packages(package.__path__, prefix="src.services.stages."):
        importlib.import_module(info.name)


def _all_subclasses(cls: type) -> set[type]:
    found: set[type] = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def _stage_output_models() -> dict[str, type[StageOutput]]:
    """Every distinct output model declared by a concrete stage handler.

    Keyed off `StageHandler.output_model` rather than "every StageOutput
    subclass": `StageOutput` is also used as a mixin on nested element models
    (DemoVideoLine et al.), which are not stage outputs and have nothing to
    produce.
    """
    _import_every_stage_module()
    models: dict[str, type[StageOutput]] = {}
    for handler in _all_subclasses(StageHandler):
        # Production stages only. Test modules define their own throwaway
        # handlers, so without this the count depends on WHICH tests have been
        # imported — the whole suite yields a different number than this file
        # alone, and the budget becomes unmeetable in one of the two.
        if not handler.__module__.startswith("src."):
            continue
        model = getattr(handler, "output_model", None)
        if isinstance(model, type) and issubclass(model, StageOutput):
            models[model.__name__] = model
    return models


def test_undeclared_stage_outputs_only_ever_shrink() -> None:
    models = _stage_output_models()
    assert models, "no stage output models discovered — the ratchet would pass vacuously"

    undeclared = sorted(name for name, m in models.items() if not m.produces)

    assert len(undeclared) <= MAX_UNDECLARED_STAGE_OUTPUTS, (
        f"{len(undeclared)} stage outputs declare no `produces` contract, budget is "
        f"{MAX_UNDECLARED_STAGE_OUTPUTS}. Declare it on the new stage rather than "
        f"raising the budget. Undeclared: {undeclared}"
    )


def test_required_producers_only_ever_grow() -> None:
    """The contract must keep the power to FAIL a run, not just annotate it.

    A `produces` declaration that is not `produces_required` cannot fail
    anything, so a codebase where the required count is zero has a contract that
    is present, tested, and unable to stop a single bad deliverable from
    shipping. That was the state until W1-1. Lowering this number is therefore
    not a refactor — it is a regression to that state, and it must be argued for
    explicitly rather than absorbed by editing a constant.
    """
    models = _stage_output_models()
    assert models, "no stage output models discovered — the ratchet would pass vacuously"

    required = sorted(name for name, m in models.items() if m.produces and m.produces_required)

    assert len(required) >= MIN_REQUIRED_STAGE_OUTPUTS, (
        f"only {len(required)} stage outputs are `produces_required`, floor is "
        f"{MIN_REQUIRED_STAGE_OUTPUTS}. Something dropped the hard-fail half of the "
        f"contract. Required: {required}"
    )


def test_a_declaring_stage_can_actually_degrade() -> None:
    """`produces` without `notes` would have nowhere to report a violation.

    The default contract degrades rather than fails, and the degrade channel is
    the `notes` list. A stage that declares `produces`, is not
    `produces_required`, and has no `notes` field would silently do nothing —
    a degrade handler that cannot fire.
    """
    offenders = [
        name
        for name, m in _stage_output_models().items()
        if m.produces and not m.produces_required and not issubclass(m, StageOutputWithNotes)
    ]
    assert not offenders, (
        "these declare `produces` but cannot degrade (no `notes` channel): "
        f"{sorted(offenders)} — subclass StageOutputWithNotes or set produces_required"
    )


def test_declared_fields_exist_on_the_model() -> None:
    """A `produces` name that is not a field is drift, and would always read empty."""
    bad: list[str] = []
    for name, model in _stage_output_models().items():
        for field in model.produces:
            if field not in model.model_fields:
                bad.append(f"{name}.{field}")
    assert not bad, f"`produces` names that are not fields on their model: {sorted(bad)}"
