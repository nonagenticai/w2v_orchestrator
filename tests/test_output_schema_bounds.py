"""CI-enforced ban on grammar-unenforceable constraint keywords in LLM output schemas.

vLLM's guided decoding constrains an LLM's output GRAMMAR and TYPE — it builds a CFG
from the JSON Schema — but a CFG cannot express "this integer is between 1 and 5".
Worker-management therefore STRIPS every such keyword
(``worker_management/src/utils/xgrammar_schema.py::strip_xgrammar_unsupported``) before
building the grammar, so a bound in an ``output_schema`` is enforced NOWHERE: it does
not stop the model emitting an out-of-range value.

It used to be worse than unenforced — it was FATAL. Validation ran against the
UNSTRIPPED schema while generation ran against the stripped one, and worker-management
sets ``retries=0`` for the ``inference_router`` provider deliberately, so the first
disagreement killed the stage permanently. Every production ``demo_video`` run failed
on 2026-07-20 at ``retention_plan:critic`` with::

    ModelRetry: Output failed schema validation. Issues:
      - at overall_score: 12 is greater than the maximum of 5

— a ``12`` the critic almost certainly echoed out of the "Deterministic retention
evidence (hard numbers)" block in its own prompt. That is the HISTORICAL mechanism, and
it is why this ban exists. It is no longer the current one: after worker_management
#141 + #144 the FATAL ``Draft202012Validator`` is built from the *stripped* schema
(``pydantic_ai_agent.py:2018`` picks ``grammar_schema`` as ``effective_grammar``,
``:2024`` compiles it), while the full schema only feeds an *advisory* validator whose
errors are logged and explicitly NOT raised (``:2088-2091``), next to a
``schema_constraints_unenforced`` signal listing the stripped keywords (``:2048-2056``).

So a stripped constraint today is INERT, not fatal — the failure mode moved from OUTAGE
to FALSE ASSURANCE, and false assurance in a schema is still a defect. An unenforced
constraint is a silent lie: the schema asserts a guarantee the grammar never applies,
so every reader of it believes a bound is being enforced when nothing enforces it.

The fix has two halves and this test guards the first: the range is DOCUMENTED in the
property's ``description`` (where the model reads it, alongside the system prompt that
already says "Score 1-5") and ENFORCED at the consumer via ``src.utils.clamp_score``.

DO NOT delete this test to make a new schema pass. Adding ``minimum``/``maximum`` back
does not give you validation — it gives you a bound nobody applies. Write the range
into ``description`` and clamp where you read the value.

Migrations are excluded, matching ``test_any_ratchet.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# The JSON Schema constraint keywords a CFG cannot express, grouped by the instance
# type they apply to. String and array constraints (``minLength``, ``minItems``,
# ``pattern``, …) are banned on exactly the same footing as the numeric bounds — this
# is NOT speculative any more. Upstream ``worker_management`` STRIPS all of these
# before building the vLLM grammar — and since #141 + #144 it validates against that
# same stripped schema, so the keyword is invisible to generation AND absent from the
# fatal validator. It is inert: a guarantee the schema advertises and nothing applies.
# (Before that parity fix the asymmetry was fatal — it took down demo_video on
# 2026-07-20 — which is why the ban was written; see the module docstring.) Every
# keyword below is stripped by ``strip_xgrammar_unsupported`` and inherits this.
#
# ``format`` is deliberately NOT banned: in JSON Schema it is annotation-only by
# default, so it neither constrains generation nor fails validation. All 7 current
# hits in the tree are video/audio format strings (``"format": "16x9"``) which the
# positional rule already excludes anyway.
# ``title`` is stripped too but likewise annotation-only, so it is not banned either.
_TYPE_FAMILIES: dict[str, frozenset[str]] = {
    "numeric": frozenset({"integer", "number"}),
    "string": frozenset({"string"}),
    "array": frozenset({"array"}),
    "object": frozenset({"object"}),
}
_KEYS_BY_FAMILY: dict[str, frozenset[str]] = {
    "numeric": frozenset(
        {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    ),
    "string": frozenset({"minLength", "maxLength", "pattern"}),
    "array": frozenset({"minItems", "maxItems"}),
    "object": frozenset({"minProperties", "maxProperties"}),
}
_BANNED_KEYS = frozenset().union(*_KEYS_BY_FAMILY.values())


def _iter_src_files() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if "/migrations/" not in p.as_posix()]


def _const_str(node: ast.expr) -> str | None:
    """The value of a string-literal node, else ``None`` (non-literal keys are opaque)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _declared_families(value: ast.expr) -> set[str]:
    """The type-families a ``"type"`` value names (numeric / string / array).

    Accepts both the scalar form (``"type": "integer"``) and the union form
    (``"type": ["integer", "null"]``). A union can span families, so the return is
    a set — a ``["integer", "string"]`` would arm both families' bans.
    """
    names: list[str] = []
    if (scalar := _const_str(value)) is not None:
        names.append(scalar)
    elif isinstance(value, (ast.List, ast.Tuple)):
        names.extend(n for elt in value.elts if (n := _const_str(elt)) is not None)
    return {family for family, types in _TYPE_FAMILIES.items() if any(n in types for n in names)}


def _violating_keys(node: ast.Dict, *, is_schema_property: bool = False) -> list[str]:
    """The banned constraint keys on a dict literal that is a schema property.

    A dict qualifies either by declaring a ``"type"`` in the SAME literal, or by
    sitting inside a ``"properties"`` mapping (``is_schema_property``). The second
    route matters: JSON Schema enforces ``minimum``/``maximum`` on a numeric
    instance even with no ``"type"`` key at all, so a bare ``{"minimum": 1}``
    property is just as fatal — and would slip a type-only check.

    A banned key is reported when EITHER the dict declares no type at all (a bare
    ``properties`` entry — every family is possible), OR it declares a type in the
    same family as the key. A ``string`` property carrying ``minimum`` is a
    nonsense combination JSON Schema ignores, so it is not flagged. Requiring one
    of the two facts keeps an ordinary ``{"minimum": 1}`` config dict, which is
    not a schema property, out of the net.
    """
    declared: set[str] = set()
    typed = False
    found: list[str] = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:  # ``{**spread}`` — no literal key to inspect
            continue
        name = _const_str(key)
        if name is None:
            continue
        if name == "type":
            fams = _declared_families(value)
            if fams:
                declared |= fams
                typed = True
        elif name in _BANNED_KEYS:
            found.append(name)

    if not found:
        return []
    # No type declared: a positional schema property could be any family, so any
    # banned key is fatal. Type declared: only bans for a matching family fire.
    if is_schema_property and not typed:
        return sorted(found)
    active = {k for fam in declared for k in _KEYS_BY_FAMILY[fam]}
    return sorted(k for k in found if k in active)


def _schema_property_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every dict literal that is a value inside a ``"properties"`` map.

    Those are schema properties by position, so their bounds are enforced regardless
    of whether they bother to declare a ``"type"``.
    """
    inside: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None or _const_str(key) != "properties":
                continue
            if not isinstance(value, ast.Dict):
                continue
            inside.update(id(v) for v in value.values if isinstance(v, ast.Dict))
    return inside


def _property_names(tree: ast.AST) -> dict[int, str]:
    """Map ``id(dict_node)`` ⇒ the string key it hangs off, for readable failures.

    ``ast`` nodes carry no parent pointer, so the mapping is built by walking every
    dict literal once and recording the key each nested dict value sits under.
    """
    named: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None or not isinstance(value, ast.Dict):
                continue
            if (name := _const_str(key)) is not None:
                named[id(value)] = name
    return named


def _scan(source: str, label: str) -> list[str]:
    """Every numeric-bound violation in one module, as human-readable strings."""
    tree = ast.parse(source)
    named = _property_names(tree)
    properties = _schema_property_nodes(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if not (keys := _violating_keys(node, is_schema_property=id(node) in properties)):
            continue
        prop = named.get(id(node), "<anonymous>")
        violations.append(f"{label}:{node.lineno} `{prop}` carries {', '.join(keys)}")
    return violations


def test_no_numeric_bounds_in_output_schemas() -> None:
    violations: list[str] = []
    for path in _iter_src_files():
        violations.extend(
            _scan(path.read_text(encoding="utf-8"), str(path.relative_to(_SRC.parent)))
        )
    assert not violations, (
        "Unenforceable schema constraint found in "
        f"{len(violations)} JSON-schema propert{'y' if len(violations) == 1 else 'ies'}:\n  "
        + "\n  ".join(violations)
        + "\n\nvLLM guided decoding constrains grammar and type only — it CANNOT enforce a "
        "numeric range, string length/pattern, array length, or object size, so these "
        "keywords do not stop a bad value being generated. worker-management STRIPS every "
        "one of them before building the grammar AND (since #141 + #144) validates against "
        "that same stripped schema, so the constraint is now INERT rather than fatal — a "
        "guarantee the schema advertises and nothing enforces. Before that parity fix the "
        "asymmetry killed 100% of prod demo_video runs on `overall_score: 12`.\n"
        "Fix: state the constraint in the property's `description` (the model reads it) and "
        "enforce it where you consume the value (e.g. `src.utils.clamp_score` for scores)."
    )


def test_detector_catches_the_shape_it_bans() -> None:
    """Prove the scan actually fires — the exact schema shape that broke prod.

    Without this, a detector bug would let the guard pass silently forever and the
    test would be decoration. Both the scalar and union ``type`` forms are exercised,
    plus the two negatives that must NOT trip it.
    """
    caught = _scan(
        """
_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": ["number", "null"], "exclusiveMinimum": 0},
    },
}
""",
        "synthetic",
    )
    assert len(caught) == 2, caught
    assert "`overall_score` carries maximum, minimum" in caught[0]
    assert "`confidence` carries exclusiveMinimum" in caught[1]

    # A bounded property with NO `"type"` key is still caught: JSON Schema enforces
    # the bound on a numeric instance regardless, so it fails validation exactly the
    # same way. Catching it needs the positional route — it sits under `properties`.
    untyped = _scan(
        '_S = {"properties": {"score": {"minimum": 1, "maximum": 5}}}',
        "synthetic",
    )
    assert len(untyped) == 1, untyped
    assert "`score` carries maximum, minimum" in untyped[0]

    # Each newly-banned string/array keyword must actually fire — otherwise widening
    # the ban is decoration. One property per keyword, each with its matching type.
    widened = _scan(
        """
_S = {
    "properties": {
        "handle": {"type": "string", "minLength": 3, "maxLength": 20},
        "slug": {"type": "string", "pattern": "^[a-z]+$"},
        "beats": {"type": "array", "minItems": 3, "maxItems": 7},
        "meta": {"type": "object", "minProperties": 1, "maxProperties": 4},
    },
}
""",
        "synthetic",
    )
    caught_keys = " ".join(widened)
    for kw in (
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        assert kw in caught_keys, f"{kw} not flagged: {widened}"

    # `format` is deliberately NOT banned — annotation-only in JSON Schema, and the
    # tree's `"format"` hits are video/audio format strings, not constraints.
    assert not _scan(
        '_FMT = {"properties": {"aspect": {"type": "string", "format": "16x9"}}}',
        "synthetic",
    )

    # A constraint on the WRONG family is a no-op JSON Schema ignores — `minimum` on
    # a string, `minLength` on an integer — so it must not be flagged.
    assert not _scan(
        """
_MISMATCH = {
    "properties": {
        "name": {"type": "string", "minimum": 1},
        "count": {"type": "integer", "minLength": 5},
    },
}
""",
        "synthetic",
    )

    # Negative: bounded properties with no bounds / non-schema config dicts.
    assert not _scan(
        """
_OK_SCHEMA = {"properties": {"overall_score": {"type": "integer", "description": "1-5"}}}
_WINDOW = {"minimum": 1, "maximum": 5, "type": "sliding"}
_LIMITS = {"minimum": 0.0, "maximum": 1.0}
""",
        "synthetic",
    )
