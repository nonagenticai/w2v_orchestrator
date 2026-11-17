"""Every background task the lifespan STARTS must also be CANCELLED on shutdown.

Audit finding 3 on the e2e-prune PR: replacing the `asgi.py` registration of
`_start_e2e_project_prune_task` with `None` left 70 tests passing — nothing
watched the wiring for that task, or for any of the other three. Driving the real
lifespan needs postgres + redis + a metrics port, so instead this reads the
lifespan's AST and asserts the invariant that actually matters: the set of
`app.state.<x>_task` attributes assigned inside `lifespan` is a subset of the
names handed to `_cancel_lifespan_tasks`.

A started-but-never-cancelled task survives the drain and keeps writing while the
pod is terminating — for this sweep, archiving rows during a rollout. The check is
deliberately whole-file rather than one hardcoded name: it costs nothing to cover
all four, it passes as written today, and a NEW task gets the guard for free.
"""

from __future__ import annotations

import ast
from pathlib import Path

ASGI = Path(__file__).resolve().parents[2] / "src" / "asgi.py"

CANCEL_HELPER = "_cancel_lifespan_tasks"


def _lifespan_node(tree: ast.Module) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            return node
    raise AssertionError("src/asgi.py has no `lifespan` coroutine — the file moved or was renamed")


def _started_task_attrs(fn: ast.AST) -> dict[str, str | None]:
    """`app.state.<name>_task = …` → the name of the function it was assigned FROM.

    The value matters, not just the attribute. An earlier version of this helper
    collected attribute names only, and `= None` still counted as "started" — so
    replacing `_start_e2e_project_prune_task(...)` with `None` (the exact audit
    finding) left this test GREEN. Found by mutation-testing this test itself.
    """
    started: dict[str, str | None] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and target.attr.endswith("_task")
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "state"
            ):
                continue
            value = node.value
            called: str | None = None
            if isinstance(value, ast.Call):
                func = value.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            # A later real assignment wins over an `= None` in an else-branch:
            # the reconciler is legitimately set to None when Redis is absent.
            if called is not None or target.attr not in started:
                started[target.attr] = called
    return started


def _cancelled_task_names(fn: ast.AST) -> set[str]:
    """String literals passed to `_cancel_lifespan_tasks(app, "a", "b", …)`."""
    cancelled: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != CANCEL_HELPER:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                cancelled.add(arg.value)
    return cancelled


def test_every_started_lifespan_task_is_also_cancelled() -> None:
    tree = ast.parse(ASGI.read_text())
    fn = _lifespan_node(tree)
    started = _started_task_attrs(fn)
    cancelled = _cancelled_task_names(fn)

    # Guard the guard: if the AST shapes ever stop matching, this test must fail
    # loudly rather than pass over two empty sets.
    assert started, "found no `app.state.*_task` assignments in lifespan — parser is stale"
    assert cancelled, f"found no `{CANCEL_HELPER}(...)` string arguments — parser is stale"

    # The sweep must be started BY ITS STARTER, not merely assigned something.
    assert started.get("e2e_project_prune_task") == "_start_e2e_project_prune_task", (
        "the e2e retention sweep is no longer started by the lifespan "
        f"(assigned from {started.get('e2e_project_prune_task')!r}); "
        "PIPELINE_E2E_PROJECT_PRUNE_* would be silently inert"
    )

    # The self-healing sound-library seed (bucket wiped 2026-10-14) must be wired.
    assert started.get("sound_library_seed_task") == "start_sound_library_seed_task", (
        "the shared sound-library startup seed is no longer started by the lifespan"
    )

    orphaned = set(started) - cancelled
    assert not orphaned, (
        "these lifespan tasks are started but never cancelled on shutdown, so they "
        f"outlive the drain and keep writing while the pod terminates: {sorted(orphaned)}"
    )
