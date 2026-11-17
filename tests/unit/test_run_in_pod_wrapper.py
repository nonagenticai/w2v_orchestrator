"""Regression: run_in_pod.build_wrapper must produce compilable source even when the
wrapped program opens with `from __future__ import ...`.

`from __future__` must be the first statement in a module, so the injected stdin/
argv preamble has to sit BELOW it — otherwise wrapping render_video.py (which uses
it) is a SyntaxError at exec time inside the pod.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "video_publish" / "run_in_pod.py"
_spec = importlib.util.spec_from_file_location("run_in_pod", _MOD_PATH)
assert _spec and _spec.loader
run_in_pod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_in_pod)


def test_wrapper_hoists_future_import_and_compiles() -> None:
    program = 'from __future__ import annotations\n"""doc"""\nimport sys\nprint(sys.argv)\n'
    wrapped = run_in_pod.build_wrapper(program, b'{"k": 1}', ["run-123"])
    # future import stays the first statement…
    assert wrapped.splitlines()[0] == "from __future__ import annotations"
    # …and the whole thing is valid Python (the original bug was a SyntaxError here).
    compile(wrapped, "<wrapped>", "exec")


def test_wrapper_leaves_future_text_inside_a_string_untouched() -> None:
    # A program that merely DOCUMENTS the pattern — the `from __future__` text lives
    # inside a docstring, not as a real import. The AST-based hoist must not relocate
    # or duplicate it: the string must survive intact and the source stay valid.
    program = (
        "from __future__ import annotations\n"
        '"""This wrapper hoists lines like\n'
        "from __future__ import annotations\n"
        'out of the body."""\n'
        "import sys\n"
        'DOC = "from __future__ import division"\n'
        "print(sys.argv, DOC)\n"
    )
    wrapped = run_in_pod.build_wrapper(program, b'{"k": 1}', ["run-9"])
    # Exactly ONE real hoisted future import at the top — the string mentions are not
    # counted, so they are neither duplicated at the top nor removed from the body.
    assert wrapped.splitlines()[0] == "from __future__ import annotations"
    assert wrapped.count("from __future__ import annotations") == 2  # 1 hoisted + 1 in docstring
    assert 'DOC = "from __future__ import division"' in wrapped
    compile(wrapped, "<wrapped>", "exec")
