"""Run-completion accounting + terminal-artifact verification.

Pure, dependency-light helpers the dispatcher's completion gate calls:

* ``terminal_stage_names`` — which stages reached a terminal state. ``skipped``
  (an opt-in stage whose precondition was absent) counts as terminal alongside
  ``completed``; if it didn't, the gate's set-equality would never hold and the
  run would hang in ``running`` forever.
* ``verify_terminal_artifact`` — the run-level backstop for the "completed but
  holds nothing" bug: even with every stage ``completed``, the deliverable must
  actually exist. WHICH field of the terminal stage's output holds the
  deliverable handle, and how to check it, are declared by the blueprint
  (``terminal_artifact_field`` / ``terminal_artifact_probe``) and enforced at
  registration — a blueprint whose terminal stage cannot supply that handle is
  rejected at import rather than completing runs with an inert probe.
"""

from __future__ import annotations

import enum
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from src.clients.design_client import design_job_exists
from src.services.tts_storage import artifact_exists

log = logging.getLogger(__name__)


class StatusRow(Protocol):
    stage_name: str
    status: str


class ArtifactRow(Protocol):
    stage_name: str

    @property
    def output(self) -> Mapping[str, Any] | None: ...


class TopoBlueprint(Protocol):
    def topological_order(self) -> Sequence[str]: ...


class ArtifactBlueprint(Protocol):
    terminal_artifact_stage: str | None

    # Which output field of that stage carries the deliverable handle, and how to
    # check it exists. Defaults live on ``DAGBlueprint``
    # (``artifact_url`` / ``object_store``); ``BlueprintRegistry.register``
    # guarantees the terminal stage actually declares the named field.
    # Read-only members on purpose: a mutable protocol attribute is INVARIANT, so
    # ``str`` would refuse ``DAGBlueprint``'s ``DeliverableProbe`` (a StrEnum).
    @property
    def terminal_artifact_field(self) -> str: ...

    @property
    def terminal_artifact_probe(self) -> str: ...


# Stage statuses that count as "this stage is done" for the run-completion gate.
TERMINAL_STAGE_STATUSES = frozenset({"completed", "skipped"})

# Terminal-BAD stage statuses: a stage that reached one of these can NEVER move
# to ``completed``/``skipped`` (all writes are write-once-guarded to the
# non-terminal set), so a run carrying one can never satisfy the DAG_COMPLETE
# set-equality and must be FAILED, not left CONTINUE-ing forever. ``cancelled``
# is only ever written by the failure sweep (``bulk_cancel_non_terminal``), so
# its presence likewise means the run is on a terminal-failure path.
FAILED_STAGE_STATUSES = frozenset({"failed", "cancelled"})


class RunDecision(enum.Enum):
    """What the dispatcher should do once a DAG-execution slice returns.

    ``DAG_COMPLETE`` is deliberately distinct from completed/failed: the
    complete-vs-fail split needs an artifact-storage probe
    (``verify_terminal_artifact``), which is I/O and stays in the dispatcher.
    This keeps ``decide_run_terminal_state`` pure and unit-testable.
    """

    CONTINUE = "continue"  # more stages pending — leave the run running
    PAUSED = "paused"  # a pause_after_stages gate was hit
    DAG_COMPLETE = "dag_complete"  # every stage terminal — verify artifact, then complete/fail
    FAILED = "failed"  # a terminal-BAD stage row exists — fail the run (never completes)


def decide_run_terminal_state(
    blueprint: TopoBlueprint, stage_rows: Sequence[StatusRow], pause_set: set[str]
) -> RunDecision:
    """Pure post-execution verdict. ``DAG_COMPLETE`` (every stage terminal —
    ``skipped`` counts) wins over ``PAUSED``; the caller must then verify the
    artifact.

    DAG_COMPLETE is checked FIRST on purpose. A genuine *fresh* pause always
    leaves downstream stages non-terminal — the executor stops at the checkpoint
    (``while remaining and not paused``) before dispatching anything that
    depends on it — so "every stage terminal" can only hold once every
    checkpoint has already been consumed. Checking ``pause_set & completed``
    first (the original ordering) re-paused a RESUMED run forever: the
    already-passed checkpoint stays ``completed`` and keeps matching
    ``pause_set`` even after the DAG has fully drained, so the run did the work
    but could never reach ``completed``. No blueprint pauses after its terminal
    stage, so letting completion win never swallows a meaningful review gate.

    ``FAILED`` is the completion gate's missing "second direction": a stage row
    in a terminal-BAD state (``failed``/``cancelled``) can never advance to
    ``completed`` (every forward write is guarded to the non-terminal set), so
    the DAG_COMPLETE set-equality can never hold. Without this branch such a run
    falls through to ``CONTINUE`` and the reconciler re-dispatches it forever
    until the ~2h max-age backstop force-fails a run that has, in memory,
    finished all its work. This happens for real when a resumable-timeout drain
    lets an independent sibling persist ``failed`` on a transient error (the
    executor does not cancel siblings on the timeout path) and that sibling then
    succeeds on resume — its success write is a guarded no-op, so the row stays
    ``failed`` and only this gate can terminalize the run. It is checked AFTER
    DAG_COMPLETE so a fully-good DAG (all completed/skipped) never trips it, and
    a deterministic failure never reaches here at all — it raises straight to the
    dispatcher's ``except Exception`` terminal path.

    ``stage_rows`` is any iterable of objects with ``.stage_name`` + ``.status``;
    ``blueprint`` need only expose ``topological_order()``.
    """
    topo = set(blueprint.topological_order())
    if topo <= terminal_stage_names(stage_rows):
        return RunDecision.DAG_COMPLETE
    if any(r.status in FAILED_STAGE_STATUSES for r in stage_rows if r.stage_name in topo):
        return RunDecision.FAILED
    completed = {r.stage_name for r in stage_rows if r.status == "completed"}
    if pause_set & completed:
        return RunDecision.PAUSED
    return RunDecision.CONTINUE


def pause_trigger_stage(
    blueprint: TopoBlueprint, stage_rows: Sequence[StatusRow], pause_set: set[str]
) -> str | None:
    """Name the stage whose completion tripped the review gate, or ``None``.

    This is the stage the human is being asked to review, and the value the
    dispatcher writes to ``runs.current_stage`` when it pauses — the FE renders
    the pause banner and builds the stage-editor href from that field.

    It is NOT the same as "the last stage that started running". ``current_stage``
    is otherwise only written at stage START (``dag_executor`` →
    ``persist(name, 'running')`` → ``mark_current_stage``), and a wave runs its
    independent stages CONCURRENTLY, so the last writer is whichever sibling
    happened to start last — frequently a no-dep leaf that then raised
    ``StageSkippedError`` and never produced anything. Leaving that value in
    place on pause sent the reviewer to a SKIPPED stage's editor (no draft),
    instead of the checkpoint they must approve.

    Mirrors ``decide_run_terminal_state``'s PAUSED predicate (``pause_set &
    completed`` — a *skipped* pause-stage never gates, see ``dag_executor``), so
    it can only return a stage that genuinely completed. With more than one
    consumed checkpoint (a re-paused raced re-dispatch), the LAST one in
    topological order wins: that is the furthest gate the run reached and the
    one it actually stopped at.
    """
    completed = {r.stage_name for r in stage_rows if r.status == "completed"}
    consumed = pause_set & completed
    if not consumed:
        return None
    ordered = [name for name in blueprint.topological_order() if name in consumed]
    # A checkpoint naming a stage outside the blueprint cannot be ordered; fall
    # back to any consumed name rather than reporting None (still better than
    # leaving a skipped stage in place).
    return ordered[-1] if ordered else sorted(consumed)[0]


def terminal_stage_names(stage_rows: Sequence[StatusRow]) -> set[str]:
    """Names of stages that reached a terminal state (completed or skipped).

    Pure helper so the "skipped counts as terminal" invariant is unit-testable
    without a database; ``stage_rows`` is any iterable of objects exposing
    ``.stage_name`` and ``.status``.
    """
    return {r.stage_name for r in stage_rows if r.status in TERMINAL_STAGE_STATUSES}


# One existence probe per ``DeliverableProbe`` kind (blueprints/base.py). Both
# share the tri-state contract: True present, False definitively absent (the ONLY
# value that may fail a run), None indeterminate. Kept as a module-level table so
# a new deliverable kind is a probe plus an enum member, and so tests can patch a
# single name.
# Thin wrappers, not the imported functions themselves: the table binds at import
# time, so storing the bare names would freeze them and defeat
# ``patch("...run_finalizer.artifact_exists")``. The wrappers resolve the global on
# each call, keeping every probe patchable at its own module-level name.
async def _probe_object_store(handle: str) -> bool | None:
    """``handle`` is a storage URL we minted — HEAD the object behind it."""
    return await artifact_exists(handle)


async def _probe_design_job(handle: str) -> bool | None:
    """``handle`` is an html_render design-job id — ask the service for its HTML."""
    return await design_job_exists(handle)


DELIVERABLE_PROBES: dict[str, Callable[[str], Awaitable[bool | None]]] = {
    "object_store": _probe_object_store,
    "design_job": _probe_design_job,
}


async def verify_terminal_artifact(
    blueprint: ArtifactBlueprint, stage_rows: Sequence[ArtifactRow]
) -> str | None:
    """Probe the blueprint's terminal deliverable; return a failure reason if it
    is *definitively* absent, else ``None`` (let the run complete).

    Even if every stage row says ``completed``, the deliverable must actually
    exist. Which output field holds the handle, and how to check it, are declared
    by the blueprint (``terminal_artifact_field`` / ``terminal_artifact_probe``) —
    NOT hard-coded here. Most blueprints publish a storage URL under
    ``artifact_url``; the two web-design blueprints publish an html_render job id
    under ``redesign_job_id``, whose user-facing links are orchestrator proxy
    routes rather than objects, so they are verified by asking html_render.

    Only a hard "it is gone" (probe → ``False``) blocks completion. A blueprint
    with no terminal stage, a skipped terminal stage (no output), or an
    indeterminate probe (``None`` — local backend, foreign host, permissions,
    transport blip) all pass: we never turn a good run red on a storage blip.

    Two things that both used to look identical (an absent probe) are now
    distinguished:

    * **A wiring gap** — the blueprint names a field the row does not carry. Since
      ``BlueprintRegistry.register`` refuses a blueprint whose terminal stage does
      not DECLARE that field, this can now only be a legacy row written before the
      field existed, or an optional field the stage left unset. It is NOT a
      verdict (a wiring gap must never fail a finished run), but it IS logged at
      ERROR: something the gate depends on is unmeasurable.
    * **A missing deliverable** — the probe ran and answered "absent". That is a
      run-level failure, exactly as before.
    """
    stage_name = blueprint.terminal_artifact_stage
    if stage_name is None:
        return None
    row = next((r for r in stage_rows if r.stage_name == stage_name), None)
    if row is None or not row.output:
        return None
    field = blueprint.terminal_artifact_field
    handle = row.output.get(field)
    if not handle:
        # WIRING GAP (not a verdict, but never silent). Before the blueprint
        # declared its field this branch also swallowed the whole class of
        # "terminal stage names its deliverable something else", which left the
        # probe permanently inert for that blueprint — `web_design_v2`'s
        # `assemble` (live_html_url / pdf_url, no artifact_url) was exactly that,
        # and its runs reported completed with the deliverable never checked.
        # Registration now rejects that shape, so reaching here means this
        # particular ROW lacks the declared handle.
        log.error(
            "terminal-artifact probe SKIPPED for stage %r: its output carries no %r, "
            "so this run's deliverable is not verified to exist",
            stage_name,
            field,
        )
        return None
    probe = DELIVERABLE_PROBES.get(str(blueprint.terminal_artifact_probe))
    if probe is None:
        log.error(
            "terminal-artifact probe SKIPPED for stage %r: unknown probe kind %r",
            stage_name,
            blueprint.terminal_artifact_probe,
        )
        return None
    try:
        exists = await probe(str(handle))
    except Exception as exc:  # noqa: BLE001 -- verifier's own error is never a run verdict
        # The probe itself failed (storage not initialised, SDK missing, …).
        # Indeterminate, not a verdict — never fail a finished run on the
        # verifier's own error.
        log.warning("terminal-artifact probe errored for run (%s=%s): %s", field, handle, exc)
        return None
    if exists is False:
        payload: dict[str, str] = {
            "reason": "artifact_verification_failed",
            "stage": stage_name,
            "field": field,
            "probe": str(blueprint.terminal_artifact_probe),
            # Historical key: consumers (and the FE) read `artifact_url`. It holds
            # the declared handle whatever the field is called.
            "artifact_url": str(handle),
        }
        return json.dumps(payload)
    return None
