"""Regression test for the AUTH_TRUST_INCLUSTER_IDENTITY opt-out in src/asgi.py.

Closes project_orchestrator_trusted_networks_bypass_2026-04-18: the orchestrator
must always fail closed, refusing to mint a synthetic in-cluster identity for
unauth external callers whose source IP happens to land in TRUSTED_NETWORKS.

The opt-out lives at the top of src/asgi.py and uses a hard assignment
(NOT setdefault) precisely so that a Helm manifest, ConfigMap, or pod env
setting AUTH_TRUST_INCLUSTER_IDENTITY=true cannot silently re-open the bypass.

This test runs a fresh subprocess so module-cache state from other tests
cannot mask a regression.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_asgi_import_overrides_preset_trust_flag_to_false() -> None:
    """Pre-set the env var to "true"; importing src.asgi must clobber it to "false".

    Also asserts that w2v_common.auth.middleware reads our authoritative value
    when imported AFTER src.asgi (the real production import order).
    """
    script = (
        "import os, sys, json\n"
        # Sanity: parent process pre-set this. Confirm it is observed by the child.
        "assert os.environ['AUTH_TRUST_INCLUSTER_IDENTITY'] == 'true', "
        "f\"setup precondition failed: {os.environ.get('AUTH_TRUST_INCLUSTER_IDENTITY')!r}\"\n"
        # Importing the orchestrator ASGI module must hard-assign the env var
        # BEFORE w2v_common.auth.middleware is imported anywhere in the process.
        "import src.asgi  # noqa: F401\n"
        "from w2v_common.auth import middleware\n"
        # Fetch the module-level constant if present. w2v_common 0.6.6+ exposes
        # _TRUST_INCLUSTER_IDENTITY; older releases do not honour the env var at
        # all -- in that case we record a sentinel and the assertion below will
        # only check the env-var hard-assignment (still load-bearing on its own).
        "flag = getattr(middleware, '_TRUST_INCLUSTER_IDENTITY', '__absent__')\n"
        "print(json.dumps({\n"
        "    'env': os.environ['AUTH_TRUST_INCLUSTER_IDENTITY'],\n"
        "    'flag': flag,\n"
        "}))\n"
    )

    env = os.environ.copy()
    env["AUTH_TRUST_INCLUSTER_IDENTITY"] = "true"
    # Don't load a real .env that might override DB/redis URLs and slow startup.
    env.setdefault("AUTH_ENABLED", "false")
    # Ensure the subprocess uses the same Python and resolves src/ from repo root.
    env["PYTHONPATH"] = str(_repo_root())

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell, trusted interpreter
        [sys.executable, "-c", script],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"subprocess failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Last line of stdout is our JSON payload (logging from import may precede it).
    payload_line = result.stdout.strip().splitlines()[-1]
    import json

    payload = json.loads(payload_line)

    assert payload["env"] == "false", (
        "src/asgi.py must hard-assign AUTH_TRUST_INCLUSTER_IDENTITY=false even "
        f"when pre-set to 'true'; got {payload['env']!r}. "
        "If this fails, setdefault has been reintroduced and the TRUSTED_NETWORKS "
        "bypass is silently re-opened."
    )
    # If the installed w2v_common exposes the constant (>=0.6.6), the
    # hard-assignment in src/asgi.py must propagate to it. Older w2v_common
    # releases don't read this env var at all; skip the downstream check rather
    # than accidentally pass an unrelated comparison.
    flag = payload["flag"]
    if flag != "__absent__":
        assert flag is False, (
            "w2v_common.auth.middleware._TRUST_INCLUSTER_IDENTITY must read "
            f"False after src.asgi import; got {flag!r}."
        )
