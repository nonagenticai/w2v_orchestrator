"""Runs routes package.

Split out of the former 1155-line ``src/api/routes/runs.py`` (read / events /
lifecycle / stages + shared helpers). The two ``APIRouter`` instances live in
``_shared``; the sub-modules decorate them and are imported below for their
registration side effects. Public symbols are re-exported so existing imports —
``from src.api.routes.runs import router | sse_router | _summary |
_project_id_for_deliverable | _tenant_owns_run | _build_run_event_snapshot |
_AUDIT_SCAN_STAGES`` — keep working.
"""

from __future__ import annotations

from ._shared import (
    _AUDIT_SCAN_STAGES,
    _project_id_for_deliverable,
    _summary,
    _tenant_owns_run,
    router,
    sse_router,
)

# Import the route sub-modules for their registration side effects (each
# decorates ``router`` / ``sse_router`` from ``_shared``).
from . import read as read  # isort: skip
from . import lifecycle as lifecycle  # isort: skip
from . import stages as stages  # isort: skip
from . import events as events  # isort: skip
from . import usage as usage  # isort: skip

# Re-exports whose canonical definition lives in a sub-module.
from .events import _build_run_event_snapshot
from .usage import usage_router

__all__ = [
    "_AUDIT_SCAN_STAGES",
    "_build_run_event_snapshot",
    "_project_id_for_deliverable",
    "_summary",
    "_tenant_owns_run",
    "router",
    "sse_router",
    "usage_router",
]
