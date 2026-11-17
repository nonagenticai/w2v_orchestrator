"""Guards on the public ``src.api.schemas`` export surface.

A model that no route declares as a ``response_model`` and no publisher ever
constructs is not "spare capacity" — it is a decoy. ``RunEvent`` was exactly
that: it claimed to be the SSE frame for ``/api/runs/{run_id}/events`` and could
not have validated a single real frame if anyone had tried. It required a
``timestamp`` no publisher emits, and it lacked the ``error`` / ``reason`` keys
that ``RunDispatcher._publish`` DOES emit. The live path is untyped JSON
published straight to the ``runs:<run_id>`` Redis channel and relayed by
``src/api/sse.py`` (see ``tests/api/test_sse_payload_shapes.py`` for what a real
frame looks like).

Deleting it is the fix; this guard is what stops it coming back. A future author
reaching for a "typed SSE frame" must model the frames publishers ACTUALLY send
— under a name that is not a known-wrong hand-mirror — rather than resurrecting
this one from an old plan doc.

``RunEventName`` (the event-NAME vocabulary in ``_shared.py``) is deliberately
NOT covered here: it is a live part of the exported surface and stays.
"""

from __future__ import annotations

from src.api import schemas


def test_run_event_model_is_not_resurrected() -> None:
    """``RunEvent`` must not exist on ``src.api.schemas`` in any form."""
    assert not hasattr(schemas, "RunEvent"), (
        "RunEvent is back on src.api.schemas. It never validated a real SSE frame "
        "(required a `timestamp` no publisher emits; missing `error`/`reason` which "
        "publishers do emit). Model the frames RunDispatcher._publish actually "
        "sends, under a new name."
    )
    assert "RunEvent" not in schemas.__all__, "RunEvent is back in src.api.schemas.__all__"


def test_run_event_name_vocabulary_is_still_exported() -> None:
    """Fence the deletion above: it must remove the dead MODEL, not the live
    event-name vocabulary that shares its prefix."""
    assert hasattr(schemas, "RunEventName")
    assert "RunEventName" in schemas.__all__


def test_schemas_all_matches_module_attributes() -> None:
    """Every name in ``__all__`` must actually resolve — a stale entry left behind
    by a deletion makes ``from src.api.schemas import *`` raise at import time."""
    missing = [name for name in schemas.__all__ if not hasattr(schemas, name)]
    assert not missing, f"names in src.api.schemas.__all__ that do not resolve: {missing}"
