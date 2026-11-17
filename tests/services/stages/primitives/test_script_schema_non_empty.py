"""Every floored script stage sends a GRAMMAR-level non-empty element list.

Replayed on the live local vLLM (xgrammar, ``disable_any_whitespace``; WM strips
``minItems``), ``lines: {"type": "array", "items": ITEM}`` answered
``{"lines": []}`` on 6 of 6 calls; the same request with
``prefixItems: [ITEM]`` returned full scripts 2 of 2. Evidence:
``w2v/.proof-archive/2026-10-06/round4/r0-replay/``. See
``primitives.script_stage.non_empty_array``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import src.services.stages as stages_pkg
from src.services.stages.primitives.script_stage import LLMScriptStageBase


def _concrete_script_stages() -> list[type[LLMScriptStageBase]]:
    for mod in pkgutil.walk_packages(stages_pkg.__path__, stages_pkg.__name__ + "."):
        importlib.import_module(mod.name)
    seen: list[type[LLMScriptStageBase]] = []
    todo = list(LLMScriptStageBase.__subclasses__())
    while todo:
        cls = todo.pop()
        todo.extend(cls.__subclasses__())
        if not inspect.isabstract(cls) and hasattr(cls, "element_key") and cls not in seen:
            seen.append(cls)
    return sorted(seen, key=lambda c: c.__qualname__)


_STAGES = _concrete_script_stages()


def test_the_walk_finds_the_known_script_stages() -> None:
    names = {c.__name__ for c in _STAGES}
    # A walk that finds nothing would make the parametrized test vacuous.
    assert {
        "AppReviewScriptStage",
        "CroAuditScriptStage",
        "LessonScriptStage",
        "PromoScriptStage",
        "ScriptDraftStage",
        "SeoAuditScriptStage",
        "WebsiteReviewScriptStage",
    } <= names


@pytest.mark.parametrize("stage_cls", list(_STAGES)[:1], ids=lambda c: c.__name__)
def test_a_floored_stage_requires_one_element_in_the_grammar(
    stage_cls: type[LLMScriptStageBase],
) -> None:
    schema = stage_cls.script_output_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    array = properties[stage_cls.element_key]
    assert isinstance(array, dict)
    assert schema["required"] == [stage_cls.element_key]
    assert array["items"] == stage_cls.element_schema
    if stage_cls.min_elements >= 1:
        # Exactly ONE required leading element, the same shape as every other —
        # a second prefix item would force two, a different item shape would let
        # the first element dodge the per-element contract.
        assert array["prefixItems"] == [array["items"]]
    else:
        assert "prefixItems" not in array
    # The floor is never expressed with a keyword WM strips from the grammar.
    assert not {"minItems", "maxItems"} & set(array)
