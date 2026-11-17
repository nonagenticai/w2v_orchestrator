"""Every poll deadline a stage hands a service client must be one the client ACCEPTS.

`BaseServiceClient.poll_job` opens with `timeout = _clamp_timeout(timeout)`, and
`_clamp_timeout` RAISES `ValueError` when the value exceeds its ceiling
(`_env_float("CLIENT_MAX_TIMEOUT_S", _MAX_TIMEOUT)`, `_MAX_TIMEOUT = 600.0`). It does
NOT silently cap. So a stage constant above the ceiling is not a tuning mistake that
merely fails to help — the poll blows up before the first GET, on every call.

⚠️ THIS IS NOT HYPOTHETICAL, AND NOTHING ELSE CATCHES IT. `demo_video/final_critique`
shipped `_ANALYSIS_POLL_TIMEOUT_S = 1100.0` (derived from its StageSpec budget, which
is the LOOSER bound and therefore the wrong binding constraint). Every unit test over
that stage passed, because they all mock the analyzer client — the real
`_clamp_timeout` never ran in any of them. The stage would have raised `ValueError`
on every analyzer poll in production.

Two properties make this test hard to fool, and both are deliberate:

1. It reads the ceiling FROM THE LIBRARY (`_clamp_timeout` / `_MAX_TIMEOUT`), never a
   hardcoded 600. If w2v_common moves the ceiling, this tracks it instead of
   asserting a number that has quietly become wrong in the other direction.
2. It finds its subjects FROM THE CALL SITES, by AST, rather than from a hand-listed
   set of constants. A THIRD stage added tomorrow with a differently-named constant
   is covered the day it is written, with no edit here. That generality is the point:
   the reason 1100.0 shipped is that no invariant connected a stage's constant to the
   transport it is spent on.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from w2v_common.clients.base import _MAX_TIMEOUT, _clamp_timeout

#: Method-name prefixes that reach `BaseServiceClient.poll_job` (directly, or via a
#: thin per-client wrapper such as `VideoAnalyzerClient.poll_analysis`), i.e. the
#: calls whose `timeout=` is clamped and can raise.
_POLLING_CALL_PREFIXES = ("poll_", "wait_for_")

_STAGES_DIR = Path("src/services/stages")


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _resolve(expr: ast.expr, module_name: str) -> float | None:
    """The value production actually sends for this `timeout=` expression.

    A bare literal resolves directly; a NAME resolves through the IMPORTED module
    attribute rather than the source text, so a constant computed from other
    constants — exactly the shape that produced 1100.0 — resolves to its real value.
    """
    if isinstance(expr, ast.Constant):
        return float(expr.value) if isinstance(expr.value, int | float) else None
    if isinstance(expr, ast.Name):
        resolved = getattr(importlib.import_module(module_name), expr.id, None)
        return float(resolved) if isinstance(resolved, int | float) else None
    return None


def _timeout_kwargs(tree: ast.AST) -> list[ast.expr]:
    """Every `timeout=` expression handed to a clamped polling call in this module."""
    return [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith(_POLLING_CALL_PREFIXES)
        for kw in node.keywords
        if kw.arg == "timeout"
    ]


def _poll_timeout_sites() -> list[tuple[str, str, float]]:
    """`(module, expression_source, resolved_value)` for every clamped poll deadline."""
    sites: list[tuple[str, str, float]] = []
    for path in sorted(_STAGES_DIR.rglob("*.py")):
        module_name = _module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for expr in _timeout_kwargs(tree):
            value = _resolve(expr, module_name)
            if value is not None:
                sites.append((module_name, ast.unparse(expr), value))
    return sites


def test_the_scan_actually_finds_the_known_poll_sites() -> None:
    """Anti-vacuity: this whole file is worthless if the AST walk silently finds nothing.

    A refactor that renames the client methods, or moves the stage tree, would empty
    the scan and turn every assertion below into a tautology over an empty list. Pin
    the two stages that motivated this guard by name.
    """
    sites = _poll_timeout_sites()
    modules = {module for module, _, _ in sites}

    assert "src.services.stages.demo_video.final_critique" in modules, (
        "the AST scan no longer sees final_critique's analyzer poll — the call-site "
        f"shape or the stage tree moved. Found: {sorted(modules)}"
    )
    assert "src.services.stages.shared.footage_critique" in modules, (
        "the AST scan no longer sees footage_critique's analyzer poll — the call-site "
        f"shape or the stage tree moved. Found: {sorted(modules)}"
    )
    assert len(sites) >= 5, (
        f"only {len(sites)} clamped poll site(s) found across {_STAGES_DIR}; the scan "
        "has stopped matching the real call shape and is close to vacuous"
    )


def test_every_stage_poll_timeout_is_accepted_by_the_client_clamp() -> None:
    """The invariant that was missing when `1100.0` shipped green.

    Asserted by CALLING the library's own `_clamp_timeout`, so this is the exact
    predicate production evaluates — not a re-implementation of it, and not a
    hardcoded 600 that would go stale if w2v_common moved the ceiling.
    """
    offenders: list[str] = []
    for module, source, value in _poll_timeout_sites():
        try:
            _clamp_timeout(value)
        except ValueError as exc:
            offenders.append(f"{module}: timeout={source} -> {value}s ({exc})")

    assert offenders == [], (
        "these stages hand a service client a poll deadline it REJECTS. "
        "`_clamp_timeout` raises ValueError above its ceiling — it does not cap — so "
        "each of these raises before the first GET on every poll, in production, "
        "while unit tests that mock the client stay green. Size the constant from the "
        f"transport ceiling ({_MAX_TIMEOUT}s), not from the StageSpec budget:\n  "
        + "\n  ".join(offenders)
    )


def test_the_ceiling_is_read_from_the_library_not_pinned_here() -> None:
    """Guard the guard: this file must track w2v_common rather than restate it.

    If `_clamp_timeout` is ever relaxed (or `CLIENT_MAX_TIMEOUT_S` is set for this
    deployment), the test above must relax with it automatically. This pins that the
    predicate really is the library's, by proving it moves when the environment moves
    the ceiling the library reads.
    """
    over_the_default = _MAX_TIMEOUT + 1.0

    with pytest.raises(ValueError, match="must be <="):
        _clamp_timeout(over_the_default)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLIENT_MAX_TIMEOUT_S", str(over_the_default + 1.0))
        assert _clamp_timeout(over_the_default) == over_the_default, (
            "`_clamp_timeout` no longer reads CLIENT_MAX_TIMEOUT_S at call time; the "
            "scan above may be asserting against a ceiling that is not the live one"
        )
