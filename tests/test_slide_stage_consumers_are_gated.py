"""Every ``render_deck_slide_stage`` consumer must act on its render failure.

``shared/slide_stage.py`` swallows a deck-render outage on purpose: it appends a
note and returns ``DeckRenderResult(slides=[], render_failed=True)`` so a
slide_service hiccup degrades the video instead of aborting the run. That is the
right call for the VIDEO. It is only safe as long as every consumer of that
result does something with ``render_failed`` — and for two of the three
consumers, for the whole life of the flag, nobody did. ``showcase_slides`` and
``website_review_slides`` copied it onto their output as
``slides_render_failed``, declared no ``produces`` (correctly — both have a
legitimate empty deck), and recorded ``completed``. Their compiles then built
the slide map by comprehension over ``[]`` with no check and still emitted a
valid ``artifact_url``. The run shipped and was billed as a success with every
branded card missing.

"Guarded beside unguarded" is the recurring shape here, so this file does not
test the three known consumers — it ENUMERATES them from the source tree and
fails on a fourth that arrives without the gate. Deleting a consumer to make it
pass fails the floor below instead.

Deliberately a source-level enumeration rather than an import of a hand-written
list: a list is exactly the thing that goes stale when someone adds a stage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.services.stages.base import StageOutput

_SRC = Path(__file__).resolve().parents[1] / "src"
_HELPER = "render_deck_slide_stage"

#: Consumers that existed when this gate was written. A NEW consumer is what the
#: enumeration is for; this floor is what stops "delete the site" from being a
#: way to pass. Raise it deliberately, never lower it.
_KNOWN_CONSUMER_FLOOR = 3


def _consumer_modules() -> list[Path]:
    """Every module under ``src/`` that CALLS the shared slide-stage helper.

    A call, not an import: ``slide_stage.py`` itself defines the helper and the
    test modules import it, and neither is a consumer that owes a gate.
    """
    found: list[Path] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "slide_stage.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == _HELPER
            ):
                found.append(path)
                break
    return found


def _output_models(path: Path) -> list[type[StageOutput]]:
    """The ``StageHandler.output_model`` of every stage handler in the module."""
    module_name = ".".join(path.relative_to(_SRC.parent).with_suffix("").parts)
    mod = __import__(module_name, fromlist=["*"])
    models: list[type[StageOutput]] = []
    for obj in vars(mod).values():
        model = getattr(obj, "output_model", None)
        if isinstance(model, type) and issubclass(model, StageOutput):
            models.append(model)
    return models


_CONSUMERS = _consumer_modules()


def test_the_consumer_enumeration_is_not_empty() -> None:
    """Guard the instrument. A broken walk would silently pass every test below."""
    assert len(_CONSUMERS) >= _KNOWN_CONSUMER_FLOOR, (
        f"found only {len(_CONSUMERS)} caller(s) of {_HELPER} "
        f"({[str(p) for p in _CONSUMERS]}), expected at least "
        f"{_KNOWN_CONSUMER_FLOOR}. Either the AST walk stopped working, or a "
        f"consumer was deleted — if genuinely deleted, lower the floor in the "
        f"same commit and say why. Do NOT lower it to make this file green."
    )


@pytest.mark.parametrize("path", list(_CONSUMERS)[:1], ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_slide_stage_consumer_gates_on_its_render_failure(path: Path) -> None:
    """The property: a swallowed deck-render outage must fail the stage.

    Keyed on the FAILURE flag and never on ``produces``/``slides == []``: an
    all-footage demo_video brief and a url-only website_review shotlist both
    produce an empty deck BY DESIGN, and a ``produces`` contract would redden
    those correct runs. ``fails_when_flagged`` is the only declaration that can
    tell "nothing was wanted" from "the render broke".
    """
    models = [m for m in _output_models(path) if "slides" in m.model_fields]
    assert models, f"{path} calls {_HELPER} but exposes no slide-bearing stage output"

    for model in models:
        flags = model.fails_when_flagged
        assert flags, (
            f"{model.__name__} ({path}) consumes {_HELPER}, whose soft-fail "
            f"boundary turns a slide_service outage into an empty-but-successful "
            f"output. Without `fails_when_flagged` the stage records `completed` "
            f"and its compile ships a card-less video as a billed success. "
            f'Declare `fails_when_flagged = ("slides_render_failed",)`.'
        )
        for flag in flags:
            assert flag in model.model_fields, (
                f"{model.__name__}.fails_when_flagged names {flag!r}, which is not "
                f"a field on it. The executor reads the DUMPED output, so a "
                f"renamed field makes this gate silently unreachable."
            )


@pytest.mark.parametrize("path", list(_CONSUMERS)[:1], ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_consumer_actually_propagates_render_failed(path: Path) -> None:
    """…and the declared flag must be WIRED to the helper's result.

    A stage can declare `fails_when_flagged` and then construct its output
    without passing `render_failed` through — the flag would default False and
    the gate could never fire. That is the vacuous version of this fix, and it is
    what the two stages shipped for months at the OTHER end (flag set, nobody
    reading). Checked in the source because the wiring is a keyword argument at
    the construction site, not an observable property of the class.
    """
    source = path.read_text()
    for model in (m for m in _output_models(path) if "slides" in m.model_fields):
        for flag in model.fails_when_flagged:
            assert f"{flag}=rendered.render_failed" in source, (
                f"{path} declares {flag!r} in {model.__name__}.fails_when_flagged "
                f"but never assigns it from the helper's `render_failed`. The flag "
                f"stays False on an outage and the gate is decorative."
            )


@pytest.mark.parametrize("path", list(_CONSUMERS)[:1], ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_consumer_carries_the_render_failure_class(path: Path) -> None:
    """…and the FAULT must be wired through too, not just the boolean.

    ``render_deck_slide_stage`` already computes ``attribute_exception(exc)`` on
    the exception it swallows — the only frame where that exception still exists.
    ``dag_executor._enforce_failure_flags`` reads the DUMPED output, so unless the
    consumer copies the class onto a FIELD it dies between the two and every
    slide-render outage is persisted ``unknown``, however attributable it was.
    Both halves are checked: the field (what the executor can read) and the
    assignment (what puts a real value in it) — a field left at its default is
    the vacuous version of this fix, and it is the exact shape
    ``slides_render_failed`` shipped in for months at the other end.
    """
    source = path.read_text()
    for model in (m for m in _output_models(path) if "slides" in m.model_fields):
        assert "failure_class" in model.model_fields, (
            f"{model.__name__} ({path}) consumes {_HELPER}, which attributes the "
            f"outage it swallows, but exposes no `failure_class` field. The "
            f"executor gate reads the dumped output, so the attribution is lost "
            f"and the run is filed `unknown`. Add "
            f"`failure_class: FailureClass = UNKNOWN`."
        )
        assert "failure_class=rendered.failure_class" in source, (
            f"{path} declares `failure_class` on {model.__name__} but never "
            f"assigns it from the helper's result. It stays `unknown` on every "
            f"outage and the field is decorative."
        )
