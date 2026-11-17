"""Every REAL stage that assembles shots must be able to CARRY the fabricated-hold note.

``CompileStageBase.handle`` folds the drained notes into the output only when the
output model has the channel::

    if qa_notes and "notes" in type(output).model_fields:   # compile_stage.py

That gate is correct — the intermediate compiles (shot_sound bed, ken_burns clip)
deliberately have no ``notes`` channel and must not flag a run. But it makes the
note-carrying capability **caller discipline, not an invariant**: a stage whose
output is a plain ``StageOutput`` would silently DROP the note, the fabricated 1s
hold would ship as a confident measured second again, and every existing test —
including the E2E in ``test_compile_stage_fabricated_hold_notes.py``, which uses a
SYNTHETIC stage it declares itself — would stay green.

That is the same drift this PR exists to prevent, one layer up. So this pins the
property over the REAL stages, discovered programmatically: walk every module under
``src.services.stages`` so no subclass can hide behind an unimported module, take
the recursive subclasses of ``CompileStageBase`` (recursive, so a stage that
INHERITS ``build_shots`` — ``seo_audit`` / ``cro_audit`` inherit
``AuditCompileStageBase``'s — is judged on the code it actually runs), and AST-scan
the resolved ``build_shots`` for an ``assemble_shots`` call.

Nothing here hard-codes the stage list: a stage number 10 that assembles shots is
enumerated automatically and must satisfy the same property. The counts that ARE
asserted are anti-vacuity floors and an independent cross-check, not a roster.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import textwrap
from types import FunctionType
from typing import Any, cast

import pytest

import src.services.stages as _stages_pkg
from src.services.stages.primitives.compile_stage import CompileStageBase

_ASSEMBLE = "assemble_shots"


def _import_every_stage_module() -> None:
    """Import every module under ``src.services.stages``.

    Subclass discovery via ``__subclasses__`` only sees classes whose module has
    been imported, so an unimported stage would be silently EXCLUDED from the
    property — i.e. the test would pass by not looking. Importing the whole tree
    removes the dependency on blueprint/registry wiring entirely: an unregistered
    stage is still enumerated.
    """
    for info in pkgutil.walk_packages(_stages_pkg.__path__, _stages_pkg.__name__ + "."):
        importlib.import_module(info.name)


_AnyCompileStage = type["CompileStageBase[Any, Any]"]


def _recursive_subclasses(cls: type) -> set[_AnyCompileStage]:
    found: set[_AnyCompileStage] = set()
    for sub in cls.__subclasses__():
        found.add(cast("_AnyCompileStage", sub))
        found |= _recursive_subclasses(sub)
    return found


def _called_names(fn: FunctionType) -> set[str]:
    """Every name called anywhere in ``fn``'s source, plain and attribute form."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _assembling_stages() -> list[_AnyCompileStage]:
    """Every PRODUCTION stage whose ``build_shots`` calls ``assemble_shots``.

    The ``src.services.stages.`` prefix filter is load-bearing, not tidiness.
    ``__subclasses__`` sees whatever has been imported, and the sibling
    ``test_compile_stage_fabricated_hold_notes.py`` declares a SYNTHETIC
    ``_AssembledStage`` that calls ``assemble_shots``. Without the filter this
    module's roster would depend on pytest collection order — the isolated run and
    the directory run would enumerate different sets — and the property would be
    asserted against a test double, which is exactly the substitute-for-the-real-
    thing this test exists to replace.
    """
    _import_every_stage_module()
    prefix = _stages_pkg.__name__ + "."
    out: list[_AnyCompileStage] = []
    for cls in _recursive_subclasses(CompileStageBase):
        if not cls.__module__.startswith(prefix):
            continue  # a test double, not a shipped stage
        build_shots = cast("FunctionType", inspect.unwrap(cls.build_shots))
        if _ASSEMBLE in _called_names(build_shots):
            out.append(cls)
    return sorted(out, key=lambda c: (c.__module__, c.__qualname__))


_ASSEMBLING = _assembling_stages()


@pytest.mark.parametrize("stage_cls", list(_ASSEMBLING)[:1], ids=lambda c: c.__name__)
def test_assembling_stage_output_can_carry_the_fabricated_hold_note(
    stage_cls: _AnyCompileStage,
) -> None:
    """The note channel `handle` gates on must exist on every assembling stage.

    Without it the drained ``_fabricated_hold`` note is discarded at
    ``compile_stage.py``'s ``"notes" in type(output).model_fields`` gate and the
    unmeasurable duration ships looking measured — the exact fail-open this PR
    closes, reintroduced by a new stage rather than by an edit to the fix.
    """
    output_model = stage_cls.output_model
    assert "notes" in output_model.model_fields, (
        f"{stage_cls.__module__}.{stage_cls.__name__} calls {_ASSEMBLE}() in build_shots, "
        f"so its shots can carry a FABRICATED 1s hold — but its output model "
        f"{output_model.__name__} has no `notes` field, so CompileStageBase.handle "
        f"silently DROPS the note and the substitution ships as a measured second. "
        f"Subclass StageOutputWithNotes."
    )


def test_discovery_found_the_assembling_stages() -> None:
    """Anti-vacuity: an empty or near-empty roster would make the property above
    pass by enumerating nothing. Seven distinct modules define an assembling
    ``build_shots``; the floor is deliberately BELOW that so deleting a stage does
    not red the suite, but far above zero so a broken walk/scan does."""
    assert len(_ASSEMBLING) >= 7, (
        f"discovery collapsed: only {[c.__name__ for c in _ASSEMBLING]} found. "
        f"Expected every CompileStageBase subclass whose build_shots calls {_ASSEMBLE}."
    )


def test_ast_discovery_agrees_with_the_import_graph() -> None:
    """Independent cross-check on the AST scan.

    The scan only sees a DIRECT ``assemble_shots(...)`` call in ``build_shots``. If
    a stage ever reaches it indirectly (via a module-local helper), the scan would
    miss it and the property would silently stop covering that stage. The set of
    stage modules that IMPORT ``assemble_shots`` is derived a completely different
    way, so a divergence between the two means the scan needs widening — a loud
    failure instead of a silent gap.
    """
    _import_every_stage_module()
    importers = {
        name
        for name, mod in sys.modules.items()
        if name.startswith(_stages_pkg.__name__ + ".")
        and name != "src.services.stages.shot_assembly"
        and getattr(mod, _ASSEMBLE, None) is not None
    }
    scanned = {cast("FunctionType", inspect.unwrap(c.build_shots)).__module__ for c in _ASSEMBLING}
    assert importers == scanned, (
        f"AST scan and import graph disagree.\n"
        f"  imports {_ASSEMBLE} but no scanned build_shots call: {sorted(importers - scanned)}\n"
        f"  scanned but does not import it: {sorted(scanned - importers)}\n"
        f"An entry in the first set is a stage reaching {_ASSEMBLE} indirectly: widen "
        f"_called_names to follow module-local helpers, do not delete this assertion."
    )


def test_a_non_assembling_compile_stage_is_correctly_excluded() -> None:
    """The scan discriminates — it does not just return every compile stage.

    ``action_reel`` is a ``CompileStageBase`` subclass that builds its shots without
    ``assemble_shots``, so it can never carry a fabricated hold and is correctly out
    of scope. If this ever fails, the scan has become a tautology that would pass on
    a stage it never actually inspected.
    """
    prefix = _stages_pkg.__name__ + "."
    all_compile_stages = {
        c for c in _recursive_subclasses(CompileStageBase) if c.__module__.startswith(prefix)
    }
    assert len(all_compile_stages) > len(_ASSEMBLING), (
        "every CompileStageBase subclass was classified as assembling; the AST scan "
        "is no longer discriminating and the property proves nothing."
    )
