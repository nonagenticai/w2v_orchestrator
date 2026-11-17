"""Prune a JSON schema to what the pgx vLLM's xgrammar backend can compile.

Vendored from worker_management ``src/utils/xgrammar_schema.py`` (@aa3b42a) for the
direct-vLLM backend (:mod:`src.services.llm.vllm_direct`), which replaces the
WM → inference-router hop that used to apply it.

Why it matters: xgrammar cannot compile several keywords (``minItems``/``maxItems``
chief among them). When it meets one it does NOT error -- it silently disables
structured decoding for the request and the model answers unconstrained. So the
grammar gets the stripped schema, and the FATAL validator is built from the same
stripped schema (WM's 2026-07-20 outage: validating against a contract the decoder
never saw failed 100% of demo_video runs). ``prefixItems`` is NOT stripped, which is
what keeps ``non_empty_array`` (orch #1076) a real grammar-level guarantee.

The walk is position-aware: keywords are dropped only in keyword position, never as
property NAMES (``title`` is a common field name) and never inside data
(``enum``/``const``/``required``).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Final

log = logging.getLogger(__name__)

XGRAMMAR_UNSUPPORTED_KEYS: Final = frozenset(
    {
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minProperties",
        "maxProperties",
        "pattern",
        "format",
        "title",
    }
)

#: Count/length bounds: checked post-hoc against the ORIGINAL schema and reported
#: (``usage_metrics.bound_violations``), never fatal -- WM's default (#358).
BOUND_KEYWORDS: Final = frozenset({"minItems", "maxItems", "minLength", "maxLength"})

_SCHEMA_MAP_KEYS: Final = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
_SUBSCHEMA_LIST_KEYS: Final = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_DATA_KEYS: Final = frozenset(
    {"enum", "const", "default", "examples", "required", "dependentRequired"}
)
_COMPOSITION_KEYS: Final = frozenset(
    {"allOf", "anyOf", "oneOf", "$ref", "if", "then", "else", "dependentSchemas"}
)


def _warn_if_unsatisfiable(original: dict[str, object], stripped: dict[str, object]) -> None:
    """Flag a stripped grammar that can no longer emit a ``required`` property."""
    props = stripped.get("properties")
    required = original.get("required")
    if not isinstance(props, dict) or not isinstance(required, list):
        return
    if any(key in original for key in _COMPOSITION_KEYS):
        return
    missing = [name for name in required if isinstance(name, str) and name not in props]
    if missing:
        log.warning(
            "xgrammar strip produced an unsatisfiable grammar: %s required but absent "
            "from properties",
            missing,
        )


def strip_xgrammar_unsupported(node: object) -> object:
    """Recursively drop xgrammar-incompatible keywords (keyword position only)."""
    if isinstance(node, list):
        return [strip_xgrammar_unsupported(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, object] = {}
    for key, value in node.items():
        if key in XGRAMMAR_UNSUPPORTED_KEYS:
            continue
        if key in _DATA_KEYS:
            out[key] = deepcopy(value)
        elif key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            out[key] = {name: strip_xgrammar_unsupported(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [strip_xgrammar_unsupported(sub) for sub in value]
        else:
            out[key] = strip_xgrammar_unsupported(value)
    _warn_if_unsatisfiable(node, out)
    return out


__all__ = ["BOUND_KEYWORDS", "XGRAMMAR_UNSUPPORTED_KEYS", "strip_xgrammar_unsupported"]
