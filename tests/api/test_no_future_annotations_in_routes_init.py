"""Trap-prevention test — see src/api/routes/__init__.py header for the why."""

import ast
from pathlib import Path


def test_routes_init_does_not_import_future_annotations() -> None:
    """Reject any `from __future__ import annotations` *statement* in the init file.

    A naive substring scan would false-positive on the documenting comment
    block at the top of the file. We parse the AST instead so only real
    import statements count.
    """
    tree = ast.parse(Path("src/api/routes/__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            names = {alias.name for alias in node.names}
            assert "annotations" not in names, (
                "src/api/routes/__init__.py must not import `annotations` from "
                "__future__ — it collides with the submodule named `annotations.py`."
            )
