"""Guard: import-time runtime deps must live in MAIN dependencies, not dev.

The production image is built with ``uv sync --frozen --no-dev`` (see
Dockerfile), so any package that is required *at import time* but declared only
in ``[dependency-groups].dev`` is absent from the running container and the app
crashloops on startup — even though every test passes locally and in CI (which
both install the dev group).

This bit us when ``python-multipart`` (needed by the audiobook manuscript
``UploadFile`` route at decoration time) sat in the dev group: #128/#129 built
green, promoted to ``:latest``, then crashlooped with
``RuntimeError: Form data requires "python-multipart" to be installed`` and the
Flux rollout timed out. CI's "Smoke pushed image" step is skipped, and unit
tests import the app with dev deps present, so nothing caught it.

This test reads pyproject.toml directly (no dev-dep masking) and asserts the
known import-time-critical packages are in ``[project].dependencies``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Packages imported (directly or by FastAPI at route-decoration time) during
# ``import src.asgi`` — i.e. required for the app to even start. Add to this set
# whenever a new import-time dependency is introduced.
_IMPORT_TIME_RUNTIME_DEPS = {
    "fastapi",
    "uvicorn",
    "python-multipart",  # FastAPI File/Form routes (uploads.py) need it at import
    "psycopg",
    "sqlalchemy",
    "pydantic",
    "redis",
}


def _project_dependency_names() -> set[str]:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    deps = data["project"]["dependencies"]
    names: set[str] = set()
    for spec in deps:
        # Strip extras and version specifiers: "psycopg[binary]>=3.3.0" -> "psycopg".
        name = spec.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("~")[0]
        names.add(name.strip().lower())
    return names


def test_import_time_deps_are_in_main_dependencies() -> None:
    declared = _project_dependency_names()
    missing = sorted(d for d in _IMPORT_TIME_RUNTIME_DEPS if d.lower() not in declared)
    assert not missing, (
        f"import-time runtime deps missing from [project].dependencies: {missing}. "
        "They must be in MAIN deps (not the dev group) or the --no-dev production "
        "image will crashloop on startup."
    )
