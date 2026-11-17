"""Contract guard for ORCH-3.2/3.3 directed-footage geometry on the wire.

The click/focus geometry recorder in ``frontend_test_runner`` emits a keyframe
bbox as a DICT (``{"x","y","w","h"}``). The compiler declares it as a 4-number
ARRAY:

    "bbox": {"prefixItems": [{"type": "number"}] * 4,
             "minItems": 4, "maxItems": 4, "type": "array"}

and ``Shot.callouts`` as ``{"items": {"$ref": ".../Callout"}, "type": "array"}``
— not a bare mapping. Both mismatches were LATENT only because the recorder's
track attachment was dropped upstream, so ``focus_track`` was null on every prod
run; the moment real geometry flows, a verbatim pass-through 422s the compile.

This module pins the emitted shape against the SAME vendored artifact the rest of
``tests/contracts`` reads (``compile_request.schema.json``, a generated copy of
``CompileRequest.model_json_schema()`` — never hand-edited), in two directions:

* the dotted-leaf walk, which catches ``focus_track.bbox.x`` as an invented field
  the compiler does not declare (a dict where an array belongs);
* a schema-driven type check of the geometry ``$defs`` themselves, which catches
  the shapes the leaf walk cannot see — a 3-element bbox, a string coordinate, or
  a mapping where an array of callouts is declared.

Stdlib only, like ``schema_walk``: ``jsonschema`` is deliberately not a dependency
of this repo.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.services.stages.shot_assembly import (
    assemble_shots,
    normalize_bbox,
    normalize_callouts,
    normalize_cursor_track,
    normalize_focus_track,
)
from tests.contracts.schema_walk import (
    as_mapping,
    load_schema,
    payload_leaves,
    schema_leaves,
    structure_violations,
    unknown_leaves,
)

_SCHEMA: Mapping[str, object] = load_schema()
_SCHEMA_LEAVES = schema_leaves(_SCHEMA)

_VIDEO = "http://v/clip.mp4"
_DICT_BBOX: dict[str, object] = {"x": 10, "y": 20, "w": 100, "h": 40}
_TUPLE_BBOX = (10.0, 20.0, 100.0, 40.0)


def _defs() -> dict[str, object]:
    defs = as_mapping(_SCHEMA["$defs"])
    assert defs is not None
    return defs


def _props(def_name: str) -> dict[str, object]:
    node = as_mapping(_defs()[def_name])
    assert node is not None, f"{def_name} missing from the vendored contract"
    props = as_mapping(node["properties"])
    assert props is not None
    return props


def _required(def_name: str) -> list[str]:
    node = as_mapping(_defs()[def_name])
    assert node is not None
    required = node.get("required", [])
    assert isinstance(required, list)
    return [str(key) for key in required]


def _assert_number(value: object, where: str) -> float:
    """Assert a payload position is a real number and hand it back narrowed."""
    assert isinstance(value, int | float) and not isinstance(value, bool), (
        f"{where}: schema declares a number, got {type(value).__name__} {value!r}"
    )
    return float(value)


def _assert_bbox_conforms(bbox: object, where: str) -> None:
    """Validate a bbox against the vendored schema's own prefixItems/min/maxItems."""
    spec = as_mapping(_props("FocusKey")["bbox"])
    assert spec is not None
    prefix = spec["prefixItems"]
    assert isinstance(prefix, list)
    assert spec["type"] == "array", "the contract stopped declaring bbox as an array"
    assert isinstance(bbox, (list, tuple)), (
        f"{where}: schema declares an array, got {type(bbox).__name__} {bbox!r} "
        "— the recorder's dict bbox reached the wire"
    )
    assert len(bbox) == spec["minItems"] == spec["maxItems"] == len(prefix), (
        f"{where}: schema declares exactly {len(prefix)} items, got {len(bbox)}"
    )
    for index, item in enumerate(bbox):
        _assert_number(item, f"{where}[{index}]")


def _assert_keyframe_conforms(key: object, def_name: str, where: str) -> None:
    """Validate one keyframe against ``$defs[def_name]``: required keys present,
    declared scalar types honoured, no key the compiler does not declare."""
    mapping = as_mapping(key)
    assert mapping is not None, f"{where}: expected an object, got {key!r}"
    props = _props(def_name)
    for name in _required(def_name):
        assert name in mapping, f"{where}: schema requires {name!r}"
    for name, value in mapping.items():
        assert name in props, f"{where}: {name!r} is not declared on {def_name}"
        spec = as_mapping(props[name])
        assert spec is not None
        declared = spec.get("type")
        if name == "bbox":
            _assert_bbox_conforms(value, f"{where}.bbox")
        elif declared == "number":
            number = _assert_number(value, f"{where}.{name}")
            minimum = spec.get("minimum")
            if isinstance(minimum, int | float):
                assert number >= minimum, f"{where}.{name}: below the schema minimum"
        elif declared == "boolean":
            assert isinstance(value, bool), f"{where}.{name}: schema declares a boolean"
        elif declared == "string":
            assert isinstance(value, str), f"{where}.{name}: schema declares a string"


def _assert_no_invented_leaves(shot: dict[str, object]) -> None:
    """The dict-bbox failure mode as the rest of tests/contracts sees it: a mapping
    where the compiler declares an array shows up as undeclared leaf positions."""
    walk = payload_leaves({"shots": [shot]})
    unknown = unknown_leaves(walk.leaves, _SCHEMA_LEAVES, scalar_leaves=walk.scalar_leaves)
    assert not unknown, f"shot carries fields the compiler does not declare: {sorted(unknown)}"
    violations = structure_violations(walk, _SCHEMA_LEAVES)
    assert not violations, f"shot puts a scalar where the compiler declares structure: {violations}"


def _built(
    *,
    focus_track_by_line: dict[str, list[dict[str, object]]] | None = None,
    cursor_track_by_line: dict[str, list[dict[str, object]]] | None = None,
    callouts_by_line: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """One footage shot off the REAL assembler, keyed by its only line."""
    shots = assemble_shots(
        storyboard_shots=[{"id": "s1", "line_id": "l1", "seconds": 4.0, "narration": "hi"}],
        images=[],
        narration_clips=[],
        video_by_line={"l1": _VIDEO},
        focus_track_by_line=focus_track_by_line,
        cursor_track_by_line=cursor_track_by_line,
        callouts_by_line=callouts_by_line,
    )
    assert len(shots) == 1
    return shots[0]


# --- the contract itself -------------------------------------------------------


def test_built_focus_track_conforms_to_vendored_focus_key_schema() -> None:
    shot = _built(
        focus_track_by_line={"l1": [{"t": 0.5, "bbox": dict(_DICT_BBOX), "action": "click"}]}
    )
    track = shot["focus_track"]
    assert isinstance(track, list) and len(track) == 1
    _assert_keyframe_conforms(track[0], "FocusKey", "shots[0].focus_track[0]")
    _assert_no_invented_leaves(shot)


def test_built_cursor_track_conforms_to_vendored_cursor_key_schema() -> None:
    shot = _built(
        cursor_track_by_line={
            "l1": [{"t": 0.0, "x": 5, "y": 5}, {"t": 0.5, "x": 60, "y": 40, "click": True}]
        }
    )
    track = shot["cursor_track"]
    assert isinstance(track, list) and len(track) == 2
    for index, key in enumerate(track):
        _assert_keyframe_conforms(key, "CursorKey", f"shots[0].cursor_track[{index}]")
    _assert_no_invented_leaves(shot)


def test_built_callouts_conform_to_vendored_callout_schema() -> None:
    shot = _built(
        callouts_by_line={
            "l1": {"type": "badge", "label": "Click", "at": 0.5, "bbox": dict(_DICT_BBOX)}
        }
    )
    callouts = shot["callouts"]
    assert isinstance(callouts, list), "Shot.callouts is declared as an array, not a mapping"
    _assert_keyframe_conforms(callouts[0], "Callout", "shots[0].callouts[0]")
    _assert_no_invented_leaves(shot)


# --- the normalizer's own rules ------------------------------------------------


def test_dict_bbox_converts_to_the_declared_tuple() -> None:
    assert normalize_bbox(_DICT_BBOX) == _TUPLE_BBOX


def test_tuple_and_list_bbox_pass_through() -> None:
    assert normalize_bbox(_TUPLE_BBOX) == _TUPLE_BBOX
    assert normalize_bbox([10, 20, 100, 40]) == _TUPLE_BBOX


def test_malformed_bbox_yields_none_rather_than_a_partial_tuple() -> None:
    assert normalize_bbox({"x": 1, "y": 2, "w": 3}) is None  # missing h
    assert normalize_bbox({"x": 1, "y": 2, "w": "wide", "h": 4}) is None  # non-numeric
    assert normalize_bbox({"x": 1, "y": 2, "w": True, "h": 4}) is None  # bool is not a size
    assert normalize_bbox([10, 20, 100]) is None  # wrong length
    assert normalize_bbox(float("nan")) is None
    assert normalize_bbox(None) is None


def test_malformed_keyframe_is_dropped_and_the_good_ones_survive() -> None:
    track = normalize_focus_track(
        [
            {"t": 0.5, "bbox": dict(_DICT_BBOX)},
            {"t": 1.0, "bbox": {"x": 1, "y": 2, "w": 3}},  # bbox unusable
            {"t": "late", "bbox": dict(_DICT_BBOX)},  # t unusable
            {"t": -1.0, "bbox": dict(_DICT_BBOX)},  # t below the schema minimum
            {"bbox": dict(_DICT_BBOX)},  # no t at all
            "not-a-keyframe",
        ]
    )
    assert track == [{"t": 0.5, "bbox": _TUPLE_BBOX}]


def test_all_keyframes_dropped_collapses_to_none_not_an_empty_list() -> None:
    assert normalize_focus_track([{"t": 0.0, "bbox": {"x": 1}}]) is None
    assert normalize_focus_track([]) is None
    assert normalize_cursor_track([]) is None
    assert normalize_callouts([]) is None


def test_absent_track_stays_absent_and_byte_identical() -> None:
    baseline = _built()
    assert "focus_track" not in baseline
    assert "cursor_track" not in baseline
    assert "callouts" not in baseline
    assert (
        _built(focus_track_by_line={"l1": []}, cursor_track_by_line={"l1": []}, callouts_by_line={})
        == baseline
    )
    # An all-malformed track is the same no-op — never an empty list on the wire.
    assert _built(focus_track_by_line={"l1": [{"t": 0.0, "bbox": {"x": 1}}]}) == baseline


def test_cursor_keys_pass_through_unchanged() -> None:
    keys: list[dict[str, object]] = [
        {"t": 0.0, "x": 5.0, "y": 5.0, "click": False},
        {"t": 0.5, "x": 60.0, "y": 40.0, "click": True},
    ]
    assert normalize_cursor_track(keys) == keys


def test_callout_with_an_unusable_bbox_is_dropped_never_guessed() -> None:
    assert normalize_callouts({"type": "badge", "label": "x", "at": 0.0, "bbox": {"x": 1}}) is None
