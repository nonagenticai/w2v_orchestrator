"""What a Restart reuses and what it re-runs — dependency-aware, per stage.

WHY. People cancel a run because something in it is wrong and they want to fix it:
they edit the script, change the voice, add a verbatim line — then Restart and
expect the work that the change does not touch to be KEPT. Before this module a
Restart either re-ran everything (any input change, any non-failed parent) or
resumed blindly from the first unfinished stage (ignoring edits made to completed
stages, so a stale narration was reused after its script changed).

The plan is a pure function of the parent run's stage rows, its snapshot, the
snapshot a new run would get NOW, its open feedback and the blueprint's stage
graph. Each stage is either REUSED (its completed output is inherited verbatim) or
RE-RUN, with the reasons. The re-run set is the union of:

* **roots** — re-run INCLUDING the stage itself:
    - a stage that did not complete in the parent (failed, cancelled mid-flight,
      never reached). Only ``completed`` rows are ever reused, so a stage cancelled
      mid-flight is never inherited from its partial output;
    - a stage with open feedback (the notes are folded into its re-run);
    - a stage that READ a run input that has changed since it ran — the keys come
      from ``stage_outputs.config_reads`` (recorded by the executor,
      ``config_reads.py``) and the live project inputs from ``STAGE_LIVE_INPUTS``.
      A row with no recorded reads (ran before 0102) re-runs on ANY input change;
* every DESCENDANT of a root (blueprint ``depends_on`` closure);
* every DESCENDANT of a stage the user EDITED after it completed (element patch /
  restamp, ``stage_outputs.edited_at``). The edited stage itself is reused — its
  edited output is the point — unless it is also a root, in which case the edit is
  reported as discarded.

Review gates follow the reuse set (``run_engine.restore_child_pause_after_stages``):
a gate on a reused stage stays approved; a gate on a re-run stage re-opens.

FULL re-run (``mode="full"``) — nothing is reused — when the blueprint cannot resume
safely (``RESUMABLE_BLUEPRINTS`` / the live-input fingerprint refuses), the parent is
not dead (``completed`` with nothing changed = "make a fresh version"), or the plan
would reuse nothing anyway.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from src.services.runs.config_reads import WHOLE_SNAPSHOT
from src.services.runs.resume_inputs import (
    RESUMABLE_BLUEPRINTS,
    RESUME_INPUTS_KEY,
    STAGE_LIVE_INPUTS,
    LiveInput,
    resume_refusal,
)
from src.services.runs.run_setup import RUN_SCOPED_SNAPSHOT_KEYS

if TYPE_CHECKING:
    import datetime

    from src.services.blueprints.base import DAGBlueprint

#: Parent states a planned Restart applies to. ``paused`` is resumed IN PLACE
#: (``POST /resume``), never re-planned; ``pending``/``running`` are refused upstream.
PLANNABLE_PARENT_STATUSES = frozenset({"failed", "cancelled", "completed"})

#: Parents whose unfinished stages are themselves roots ("pick up where it stopped").
_DEAD_PARENT_STATUSES = frozenset({"failed", "cancelled"})

Action = Literal["reuse", "rerun"]

#: Machine codes for a :class:`Reason`. The Studio renders each one in its own words
#: (stage + input labels); ``message`` is the plain-English fallback.
ReasonCode = Literal[
    # re-run — the stage itself
    "cancelled",  # cancelled mid-flight in the parent: partial work discarded
    "failed",
    "not_run",  # the parent never reached it
    "skipped",  # re-evaluates its own opt-in gate
    "missing_output",
    "input_changed",  # ``inputs`` = the changed keys it read
    "inputs_unrecorded",  # ran before read-recording; an input changed
    "reads_everything",  # read the whole snapshot; an input changed
    "live_input_changed",  # ``inputs`` = live project data it reads
    "feedback",  # open feedback to apply
    # re-run — because of another stage
    "upstream_rerun",  # ``stage`` re-runs and this depends on it
    "upstream_edited",  # ``stage`` was edited and this depends on it
    "edit_replaced",  # this stage was edited, but it re-runs anyway
    # reuse
    "unchanged",
    "edit_kept",
    # whole plan is a full re-run
    "not_plannable",
    "not_resumable",
    "nothing_changed",
    "all_steps_finished",  # a dead run whose every step completed
    "nothing_reusable",
]


@dataclass(frozen=True)
class Reason:
    code: ReasonCode
    message: str
    stage: str | None = None
    inputs: tuple[str, ...] = ()


class StageRow(Protocol):
    """The slice of a ``stage_outputs`` row the planner reads (read-only)."""

    @property
    def stage_name(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def output(self) -> object: ...
    @property
    def completed_at(self) -> datetime.datetime | None: ...
    @property
    def edited_at(self) -> datetime.datetime | None: ...
    @property
    def config_reads(self) -> list[str] | None: ...


@dataclass(frozen=True)
class StagePlan:
    stage_name: str
    action: Action
    reasons: tuple[Reason, ...]
    #: The user edited this stage's output after it completed.
    edited: bool = False


@dataclass(frozen=True)
class RestartPlan:
    mode: Literal["partial", "full"]
    stages: tuple[StagePlan, ...]
    #: Why nothing is reused (``mode="full"``); empty for a partial plan.
    full_reasons: tuple[Reason, ...] = ()
    #: Run inputs that differ from the parent's (top-level snapshot keys, plus
    #: ``live:<input>`` for project data a stage reads live).
    changed_inputs: tuple[str, ...] = ()
    #: Re-run INCLUDING the stage (only meaningful for ``mode="partial"``).
    rerun_roots: frozenset[str] = field(default_factory=frozenset)
    #: Edited stages: reused themselves, their descendants re-run.
    edited_stages: frozenset[str] = field(default_factory=frozenset)

    @property
    def reused(self) -> list[str]:
        return [s.stage_name for s in self.stages if s.action == "reuse"]

    @property
    def rerun(self) -> list[str]:
        return [s.stage_name for s in self.stages if s.action == "rerun"]


def _canon(value: object) -> str:
    return json.dumps(json.loads(json.dumps(value, default=str)), sort_keys=True)


def changed_snapshot_keys(parent: Mapping[str, object], would_be: Mapping[str, object]) -> set[str]:
    """Top-level order keys whose value differs (added or removed keys included).

    Same exclusions as ``order_fingerprint`` (request/run-scoped identity keys), plus
    the live-input fingerprint, which is compared per input by
    :func:`changed_live_inputs`.
    """
    keys = (set(parent) | set(would_be)) - RUN_SCOPED_SNAPSHOT_KEYS - {RESUME_INPUTS_KEY}
    return {
        k
        for k in keys
        if (k in parent) != (k in would_be) or _canon(parent.get(k)) != _canon(would_be.get(k))
    }


def changed_live_inputs(
    parent: Mapping[str, object], would_be: Mapping[str, object]
) -> set[LiveInput]:
    """Live project inputs whose fingerprint moved. A version change moves all."""
    p = parent.get(RESUME_INPUTS_KEY)
    w = would_be.get(RESUME_INPUTS_KEY)
    if not isinstance(p, Mapping) or not isinstance(w, Mapping):
        return set(LiveInput)
    if p.get("version") != w.get("version"):
        return set(LiveInput)
    return {li for li in LiveInput if _canon(p.get(li.value)) != _canon(w.get(li.value))}


def _input_reasons(
    row: StageRow, changed_keys: set[str], changed_live: set[LiveInput]
) -> list[Reason]:
    reasons: list[Reason] = []
    reads = row.config_reads
    if changed_keys and reads is None:
        reasons.append(
            Reason(
                "inputs_unrecorded",
                "a project setting changed and this step ran before its inputs were "
                "recorded, so it re-runs to be safe",
                inputs=tuple(sorted(changed_keys)),
            )
        )
    elif changed_keys and reads is not None and WHOLE_SNAPSHOT in reads:
        reasons.append(
            Reason(
                "reads_everything",
                "it reads every project setting, and one changed",
                inputs=tuple(sorted(changed_keys)),
            )
        )
    elif reads is not None and (hit := sorted(changed_keys & set(reads))):
        reasons.append(
            Reason("input_changed", f"it uses {', '.join(hit)}, which changed", inputs=tuple(hit))
        )
    live = STAGE_LIVE_INPUTS.get(row.stage_name, frozenset())
    if live_hit := sorted(li.value for li in changed_live & live):
        reasons.append(
            Reason(
                "live_input_changed",
                f"it uses the project's {', '.join(live_hit)}, which changed",
                inputs=tuple(live_hit),
            )
        )
    return reasons


def _not_finished_reason(row: StageRow | None) -> Reason:
    status = row.status if row is not None else "pending"
    if status == "completed":
        return Reason("missing_output", "its saved result is missing, so it cannot be reused")
    if status == "cancelled":
        return Reason("cancelled", "it was cancelled before it finished; partial work is discarded")
    if status == "failed":
        return Reason("failed", "it failed last time")
    if status == "skipped":
        return Reason("skipped", "it was skipped last time and is checked again")
    return Reason("not_run", "it had not run yet")


def _reusable(row: StageRow | None) -> bool:
    return row is not None and row.status == "completed" and row.output is not None


def _edited_after_completion(row: StageRow) -> bool:
    if row.edited_at is None:
        return False
    return row.completed_at is None or row.edited_at >= row.completed_at


def _full(order: Sequence[str], reason: Reason, changed: Sequence[str] = ()) -> RestartPlan:
    return RestartPlan(
        mode="full",
        stages=tuple(StagePlan(n, "rerun", (reason,)) for n in order),
        full_reasons=(reason,),
        changed_inputs=tuple(changed),
    )


def _refuse_plan(
    *,
    blueprint: DAGBlueprint,
    parent_status: str,
    parent_snapshot: Mapping[str, object],
    would_be_snapshot: Mapping[str, object],
) -> Reason | None:
    """Why no stage may be reused at all, or ``None`` when a plan is possible."""
    if parent_status not in PLANNABLE_PARENT_STATUSES:
        return Reason("not_plannable", f"a {parent_status} run is not restarted from a plan")
    if blueprint.id not in RESUMABLE_BLUEPRINTS:
        return Reason("not_resumable", "this kind of project re-runs every step on restart")
    refusal = resume_refusal(parent_snapshot) or resume_refusal(would_be_snapshot)
    if refusal:
        return Reason("not_resumable", "; ".join(refusal))
    return None


def _classify(
    *,
    order: Sequence[str],
    by_name: Mapping[str, StageRow],
    parent_status: str,
    changed_keys: set[str],
    changed_live: set[LiveInput],
    annotated_stages: Collection[str],
) -> tuple[dict[str, list[Reason]], set[str]]:
    """``(roots → their own reasons, edited stages)``."""
    roots: dict[str, list[Reason]] = {}
    edited: set[str] = set()
    for name in order:
        row = by_name.get(name)
        if row is not None and row.status == "skipped":
            # A skip is a decision taken from the inputs (an absent opt-in block). It
            # re-checks its own gate cheaply either way; it drags its dependents along
            # only when an input it read changed — e.g. AI images were switched ON.
            if skip_reasons := _input_reasons(row, changed_keys, changed_live):
                roots[name] = skip_reasons
            continue
        if row is None or not _reusable(row):
            # Not reusable whatever else holds. On a DEAD parent this is where it
            # stopped; on a completed parent nothing unfinished is a root (there is
            # nothing to resume).
            if parent_status in _DEAD_PARENT_STATUSES:
                roots[name] = [_not_finished_reason(row)]
            continue
        reasons = _input_reasons(row, changed_keys, changed_live)
        if name in annotated_stages:
            reasons.append(Reason("feedback", "it has open feedback to apply"))
        if reasons:
            roots[name] = reasons
        if _edited_after_completion(row):
            edited.add(name)
    return roots, edited


def _stage_plans(
    *,
    blueprint: DAGBlueprint,
    order: Sequence[str],
    by_name: Mapping[str, StageRow],
    roots: Mapping[str, list[Reason]],
    edited: set[str],
) -> list[StagePlan]:
    caused_by: dict[str, list[Reason]] = {}
    for root in roots:
        for d in blueprint.descendants(root):
            caused_by.setdefault(d, []).append(
                Reason("upstream_rerun", f"it depends on {root}, which re-runs", stage=root)
            )
    for e in sorted(edited):
        for d in blueprint.descendants(e):
            caused_by.setdefault(d, []).append(
                Reason("upstream_edited", f"it depends on your edit to {e}", stage=e)
            )
    plans: list[StagePlan] = []
    for name in order:
        row = by_name.get(name)
        is_edited = name in edited
        reasons = [*roots.get(name, []), *caused_by.get(name, [])]
        if reasons:
            if is_edited:
                reasons.append(
                    Reason("edit_replaced", "your edits to this step are replaced by the re-run")
                )
            plans.append(StagePlan(name, "rerun", tuple(reasons), edited=is_edited))
        elif _reusable(row):
            why = (
                Reason("edit_kept", "your edited version is kept")
                if is_edited
                else Reason("unchanged", "nothing it depends on changed")
            )
            plans.append(StagePlan(name, "reuse", (why,), edited=is_edited))
        else:
            plans.append(StagePlan(name, "rerun", (_not_finished_reason(row),)))
    return plans


def compute_restart_plan(
    *,
    blueprint: DAGBlueprint,
    parent_status: str,
    rows: Sequence[StageRow],
    parent_snapshot: Mapping[str, object],
    would_be_snapshot: Mapping[str, object],
    annotated_stages: Collection[str],
) -> RestartPlan:
    """The per-stage reuse/re-run plan for a bare Restart of the parent. Pure."""
    order = blueprint.topological_order()
    refused = _refuse_plan(
        blueprint=blueprint,
        parent_status=parent_status,
        parent_snapshot=parent_snapshot,
        would_be_snapshot=would_be_snapshot,
    )
    if refused is not None:
        return _full(order, refused)

    changed_keys = changed_snapshot_keys(parent_snapshot, would_be_snapshot)
    changed_live = changed_live_inputs(parent_snapshot, would_be_snapshot)
    changed = sorted(changed_keys) + sorted(f"live:{li.value}" for li in changed_live)
    by_name = {r.stage_name: r for r in rows}
    roots, edited = _classify(
        order=order,
        by_name=by_name,
        parent_status=parent_status,
        changed_keys=changed_keys,
        changed_live=changed_live,
        annotated_stages=annotated_stages,
    )
    if not roots and not edited:
        # Nothing would re-run, so a "partial" child would just copy the parent.
        # On a completed run that is a request for a fresh take; on a failed one
        # every step finished and the failure came after them (e.g. the final
        # artifact was missing) — copying them back would reproduce it.
        if parent_status == "completed":
            why = Reason(
                "nothing_changed",
                "nothing changed since this run finished, so a new version re-runs every step",
            )
        else:
            why = Reason(
                "all_steps_finished",
                "every step had finished; the run stopped after them, so every step runs again",
            )
        return _full(order, why, changed)
    plans = _stage_plans(
        blueprint=blueprint, order=order, by_name=by_name, roots=roots, edited=edited
    )
    if not any(p.action == "reuse" for p in plans):
        return _full(order, Reason("nothing_reusable", "no finished step can be reused"), changed)
    return RestartPlan(
        mode="partial",
        stages=tuple(plans),
        changed_inputs=tuple(changed),
        rerun_roots=frozenset(roots),
        edited_stages=frozenset(edited),
    )
