#!/usr/bin/env python3
"""Run an in-cluster helper (render_video.py / host_package.py) inside a
w2v-orchestrator pod, codifying the base64-embed exec pattern.

Why this exists: the w2v orchestrator API is in-cluster only, so these helpers
must run from inside a pod via ``kubectl exec``. Two gotchas the README used to
describe only in prose (now codified here):

  1. You cannot ``python - <<HEREDOC`` AND pipe data — the heredoc and the data
     pipe both claim stdin and the data is lost. So this EMBEDS the program's
     stdin as a base64 literal inside a small wrapper and pipes the whole wrapper
     to ``python -``; the wrapper re-exposes the bytes as ``sys.stdin``.
  2. Pods are multi-replica (HPA); ``/tmp`` is not shared across execs. Each
     invocation is one self-contained exec — no staging step to race.

Secrets: pass config via ``--env`` for NON-secret values only (they land in the
pod process argv). The render token is already the pod's own
``$DOWNSTREAM_LICENSE_TOKEN``; a demo password belongs in the brief JSON (stdin),
never in ``--env``.

Examples
--------
Render (brief on stdin), print the exec without running it first:
    python run_in_pod.py --program render_video.py --stdin brief.json --print

Host a recording-package zip (base64 the zip in, capture PACKAGE_URL):
    python run_in_pod.py --program host_package.py --stdin recording-package.zip \\
        --env VIDEO_PKG_FILENAME=my-app-recording.zip
"""

from __future__ import annotations

import argparse
import ast
import base64
import shlex
import subprocess
import sys
from pathlib import Path

# No cluster access path ships in the published source tree. Override with
# --kubectl (any kubectl prefix, including an ssh wrapper, is accepted).
DEFAULT_KUBECTL = "kubectl"
DEFAULT_NAMESPACE = "w2v"
DEFAULT_SELECTOR = "app=w2v-orchestrator"
DEFAULT_CONTAINER = "w2v-orchestrator"


def _future_import_linenos(program_src: str) -> set[int]:
    """1-indexed source line numbers of the program's real ``from __future__`` imports.

    AST-based, NOT a text scan: a line reading ``from __future__ import ...`` inside a
    string/docstring is not an ``ImportFrom`` node, so it is left in the body untouched
    (a text scan would yank it out and corrupt the literal). An unparseable program
    yields the empty set — nothing to hoist, and wrapping it would fail regardless.
    """
    try:
        tree = ast.parse(program_src)
    except SyntaxError:
        return set()
    linenos: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            linenos.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return linenos


def build_wrapper(program_src: str, stdin_bytes: bytes | None, prog_argv: list[str]) -> str:
    """Wrap ``program_src`` so it runs with ``stdin_bytes`` as its stdin and
    ``prog_argv`` as its argv — all self-contained (no heredoc, no staging)."""
    preamble = ["import sys"]
    if prog_argv:
        preamble.append(f"sys.argv = {['prog', *prog_argv]!r}")
    if stdin_bytes is not None:
        b64 = base64.b64encode(stdin_bytes).decode("ascii")
        preamble += [
            "import base64, io",
            f'_DATA = base64.b64decode("{b64}")',
            "sys.stdin = io.TextIOWrapper(io.BytesIO(_DATA))",
        ]
    # `from __future__` imports must be the first statement in the module, so hoist
    # any out of the program above the injected preamble — otherwise wrapping a
    # program that uses them (e.g. render_video.py) is a SyntaxError. Identify them
    # by AST so a ``from __future__`` line inside a docstring/string isn't relocated.
    hoist = _future_import_linenos(program_src)
    lines = program_src.splitlines()
    future = [ln for i, ln in enumerate(lines, start=1) if i in hoist]
    body = "\n".join(ln for i, ln in enumerate(lines, start=1) if i not in hoist)
    return "\n".join([*future, *preamble]) + "\n\n" + body


def resolve_pod(kubectl: list[str], namespace: str, selector: str) -> str:
    cmd = [
        *kubectl,
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        selector,
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ]
    # trusted list-argv (kubectl + operator-supplied flags), no shell — safe
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    pod = out.stdout.strip()
    if not pod:
        raise SystemExit(f"no Running pod matched -l {selector} in {namespace}")
    return pod


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--program", required=True, help="in-pod python program (e.g. render_video.py)")
    ap.add_argument("--stdin", help="file whose bytes become the program's stdin (base64-embedded)")
    ap.add_argument(
        "--env", action="append", default=[], metavar="K=V", help="non-secret env for the program"
    )
    ap.add_argument("--prog-arg", action="append", default=[], help="argv passed to the program")
    ap.add_argument("--kubectl", default=DEFAULT_KUBECTL, help="ssh+kubectl prefix")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--selector", default=DEFAULT_SELECTOR)
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--pod", help="explicit pod name (skips label resolution)")
    ap.add_argument(
        "--print", dest="print_only", action="store_true", help="assemble + print, do not exec"
    )
    args = ap.parse_args()

    program_src = Path(args.program).read_text(encoding="utf-8")
    stdin_bytes = Path(args.stdin).read_bytes() if args.stdin else None
    wrapper = build_wrapper(program_src, stdin_bytes, args.prog_arg)

    # No shell runs this (subprocess list form), so expand ~ ourselves — ssh does
    # not expand `~` in `-i ~/.ssh/key`.
    kubectl = [Path(t).expanduser().as_posix() for t in shlex.split(args.kubectl)]
    env_prefix = ["env", *args.env] if args.env else []

    if args.print_only:
        pod = args.pod or f"<pod -l {args.selector}>"
        exec_cmd = [
            *kubectl,
            "exec",
            "-i",
            "-n",
            args.namespace,
            pod,
            "-c",
            args.container,
            "--",
            *env_prefix,
            "python",
            "-",
        ]
        print("# wrapper piped to stdin (first 400 chars):", file=sys.stderr)
        print(wrapper[:400] + ("…" if len(wrapper) > 400 else ""), file=sys.stderr)
        print("# exec command:")
        print(shlex.join(exec_cmd))
        return 0

    pod = args.pod or resolve_pod(kubectl, args.namespace, args.selector)
    exec_cmd = [
        *kubectl,
        "exec",
        "-i",
        "-n",
        args.namespace,
        pod,
        "-c",
        args.container,
        "--",
        *env_prefix,
        "python",
        "-",
    ]
    # trusted list-argv (kubectl + operator-supplied flags), no shell — safe
    proc = subprocess.run(exec_cmd, input=wrapper, text=True)  # noqa: S603
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
