"""ONE definition of the directed-footage geometry WIRE shape, for tests.

The click/focus recorder in ``frontend_test_runner`` emits a keyframe bbox as a
DICT (``{"x","y","w","h"}``) plus a recorder-only ``action`` key. The compiler
declares ``FocusKey.bbox`` / ``Callout.bbox`` as a 4-number ARRAY and declares no
``action`` (see ``tests/contracts/compile_request.schema.json``), so
``shot_assembly._attach_clip_framing`` converts at that boundary.

Several test modules assert the converted shape. These helpers are the single
place that spells it out, so the wire contract is stated once rather than
hand-copied as literal tuples in five files.

DELIBERATELY independent of ``src``: they restate the contract from the vendored
schema by hand rather than calling ``normalize_bbox`` & friends. Importing the
implementation would make every assertion that uses them compare the code to
itself — vacuous, and green against any regression.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

BboxWire = tuple[float, float, float, float]


def _num(raw: object) -> float:
    assert isinstance(raw, (int, float)) and not isinstance(raw, bool), (
        f"expected a recorder coordinate, got {raw!r}"
    )
    return float(raw)


def _mapping(raw: object) -> Mapping[str, object]:
    assert isinstance(raw, Mapping), f"expected a recorder bbox mapping, got {raw!r}"
    return cast("Mapping[str, object]", raw)


def bbox_wire(bbox: Mapping[str, object]) -> BboxWire:
    """A recorder ``{"x","y","w","h"}`` bbox as the compiler's ``(x, y, w, h)``."""
    return (_num(bbox["x"]), _num(bbox["y"]), _num(bbox["w"]), _num(bbox["h"]))


def focus_wire(track: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """A recorder focus track as the compiler's ``list[FocusKey]``: same keyframes,
    same order, same count — ``bbox`` as a 4-tuple and ``action`` dropped."""
    return [{"t": _num(key["t"]), "bbox": bbox_wire(_mapping(key["bbox"]))} for key in track]


def cursor_wire(track: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """A recorder cursor track as the compiler's ``list[CursorKey]``. The recorder
    shape already matches; only the numbers are floats on the wire, and ``click`` is
    carried through when the recorder set it."""
    out: list[dict[str, object]] = []
    for key in track:
        wire: dict[str, object] = {"t": _num(key["t"]), "x": _num(key["x"]), "y": _num(key["y"])}
        click = key.get("click")
        if isinstance(click, bool):
            wire["click"] = click
        out.append(wire)
    return out


def callouts_wire(callout: Mapping[str, object]) -> list[dict[str, object]]:
    """One directed-plan callout as the compiler's ``Shot.callouts`` ARRAY."""
    return [{**callout, "bbox": bbox_wire(_mapping(callout["bbox"]))}]
