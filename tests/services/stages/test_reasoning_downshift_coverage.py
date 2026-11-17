"""Every WM call must decide, explicitly, what it does with reasoning on a RETRY.

The failure this pins was measured in production on 2026-08-20: with
chain-of-thought ON the model burned its whole 950s deadline and returned nothing,
so ``stage_call``'s retry — an identical request against an identical budget — was
guaranteed to fail identically. Four paid runs died that way in one day.

A new ``WMClient.message`` call site that says nothing about reasoning inherits that
ladder silently. So the gate is not "did you call the helper" (a site can legitimately
hardcode ``enable_reasoning=False``, as #883 did on the two measured stages) but "is
the decision written down at the call site at all".
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[3] / "src"

# A WM message call that does not name `enable_reasoning`. Zero today. This may only
# ever go DOWN: a new site must say what it does, in one keyword.
MAX_UNDECIDED_WM_CALLS = 0


def _undecided_wm_calls() -> list[str]:
    out: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "message":
                continue
            kwargs = {kw.arg for kw in node.keywords}
            # `worker_name` is what distinguishes a WMClient.message from any other
            # .message() in the tree (logging, redis pubsub, ...).
            if "worker_name" not in kwargs:
                continue
            if "enable_reasoning" not in kwargs:
                out.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
    return out


def test_no_wm_call_leaves_reasoning_undecided() -> None:
    undecided = _undecided_wm_calls()
    assert len(undecided) <= MAX_UNDECIDED_WM_CALLS, (
        f"{len(undecided)} WM call(s) do not say what they do with reasoning "
        f"(budget {MAX_UNDECIDED_WM_CALLS}). Pass `enable_reasoning=` — "
        f"`reasoning_for_attempt()` inside a loop_engine closure, `think` from a "
        f"`downshift_reasoning` factory, or a hardcoded value if the stage was "
        f"MEASURED thinking-off. Sites: {undecided}"
    )


def test_the_scan_is_not_vacuous() -> None:
    """A scanner that finds no WM calls at all would pass the ratchet forever."""
    found = 0
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "message"
                and any(kw.arg == "worker_name" for kw in node.keywords)
            ):
                found += 1
    assert found >= 30, f"only {found} WM call sites found — the scanner stopped seeing them"
