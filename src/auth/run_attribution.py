"""Carry a run's CALLER attribution (actor + fr project) into its background task.

w2v-common 0.88 uses ``SERVICE_ACTOR_ID`` only for SERVICE-originated work. A run
executes in a background task (fresh dispatch, resume, reconciler re-dispatch,
reaper, reconciler-finalize) where ``set_identity_context`` clears the actor and
the fr project, and — on the reconciler paths — no inbound request exists, so
w2v_common would stamp the orchestrator's OWN actor on every stage's cost events
and outbound calls. A run is CALLER-originated work, so that is wrong.

The caller's actor (canonical UUID) and fr project are snapshotted into
``config_snapshot`` at run creation, in request scope, and re-bound after every
``set_identity_context`` that rebinds identity from the snapshot.

A run with no stored actor (a legacy row, or a caller that sent none) stays
ACTOR-LESS and is counted by w2v_common as ``inbound_without_actor``. It never
falls back to SERVICE_ACTOR_ID: the run was caller-originated, and substituting
the orchestrator would hide the caller's gap from billing coverage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from w2v_common.auth import (
    ACTOR_ID_HEADER,
    PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    get_project_id_from_context,
)
from w2v_common.auth.identity_context import mark_inbound_request, resolve_actor_id

__all__ = [
    "ACTOR_ID_KEY",
    "ATTRIBUTION_PROJECT_KEY",
    "bind_run_attribution",
    "snapshot_run_attribution",
]

logger = logging.getLogger(__name__)

#: ``config_snapshot`` key for the caller's actor (canonical UUID, "" when none).
ACTOR_ID_KEY = "actor_id"
#: ``config_snapshot`` key for the caller's FR project (``X-Project-Id``). Distinct
#: from the snapshot's ``project_id``, which is the W2V project.
ATTRIBUTION_PROJECT_KEY = "attribution_project_id"


def snapshot_run_attribution() -> dict[str, str]:
    """The caller's actor + fr project, read in REQUEST scope at run creation."""
    actor_id, _missing = resolve_actor_id()
    return {
        ACTOR_ID_KEY: actor_id or "",
        ATTRIBUTION_PROJECT_KEY: get_project_id_from_context(),
    }


def bind_run_attribution(snapshot: Mapping[str, object]) -> None:
    """Re-bind the snapshotted actor + fr project. Call AFTER ``set_identity_context``.

    Marks the context inbound (caller-originated) so a missing actor is counted, not
    replaced with SERVICE_ACTOR_ID. Values go through w2v_common's public
    ``bind_attribution_from_headers`` (trusted: captured from this service's own
    validated request context), so a malformed stored value is ignored, never coerced.
    """
    mark_inbound_request()
    actor = str(snapshot.get(ACTOR_ID_KEY) or "")
    project = str(snapshot.get(ATTRIBUTION_PROJECT_KEY) or "")
    headers = {ACTOR_ID_HEADER: actor, PROJECT_ID_HEADER: project}
    bind_attribution_from_headers({k: v for k, v in headers.items() if v}, trusted=True)
    if not actor:
        logger.info(
            "run snapshot has no caller actor; running actor-less "
            "(counted as inbound_without_actor, no SERVICE_ACTOR_ID fallback)"
        )
