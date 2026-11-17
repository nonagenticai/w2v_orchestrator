"""RunContext — the immutable bag of state passed to every StageHandler.

A RunContext is built once at the start of a run by RunEngine. Stage
handlers receive it as the second argument to ``handle``. They can read
``prior_run_outputs`` and ``open_annotations_by_stage`` (notes from the
parent run that should be folded into the prompt for this stage).
Handlers MUST NOT mutate the context.

⚠️ ``prior_run_outputs`` is **NOT regenerate-only** — this docstring said so
until 2026-08-15 and it is the kind of wrong that costs a render. It is the
ORDINARY within-run channel by which a stage reads an EARLIER stage's output:
``dag_executor.run`` seeds it from the run's persisted stage_outputs and then
``replace(ctx, prior_run_outputs={**outputs})`` re-builds it before every
stage, so by the time a stage runs it holds every completed output of THIS run
— filtered by nothing, not by ``depends_on``. Believing the old line, a reader
would delete "unreachable" reads like ``ctx.prior_output("footage_probe")`` in
``action_reel``'s cadence/cut_plan/compile chain and silently drop three stages
back to the 30 fps default. A transitive dependency is enough for the output to
be there; a stage that wants the value must simply be ordered after its producer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from src.services.stages.shared.step_ledger import StepLedger


#: Snapshot key a child run records the stage the CALLER explicitly asked to re-run
#: from (``POST /runs/{id}/regenerate {"from_stage": ...}``). Absent on derived cuts
#: (failed-parent restart, structural relief, music rewind). A stage reads it to tell
#: "re-run me, I want new output" from "re-run because something upstream moved".
REGENERATE_FROM_STAGE_KEY = "regenerate_from_stage"


@dataclass(frozen=True)
class RunContext:
    run_id: UUID
    deliverable_id: UUID
    config_snapshot: dict[str, Any]
    prior_run_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_annotations_by_stage: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tenant_id: UUID | None = None
    subject_id: UUID | None = None
    correlation_id: str = ""
    # Distinguishes DISTINCT renders that legitimately share a run_id + stage
    # name — a footage-critique loop re-drives the SAME compile stage with nudged
    # shots on a ``replace(ctx, ...)``. The compile idempotency key folds this in
    # so a critique re-render is NOT deduped to the original render (the empty
    # default keeps a normal render's key stable across a resumable-timeout
    # re-dispatch, so its re-submit dedupes). NOTE: this only makes the re-render's
    # key DISTINCT; it is stable across a resume only if the value itself is stable.
    # A discriminator derived from a non-deterministic source (e.g. a re-computed
    # LLM critique verdict) does NOT make the re-render's OWN resume dedupe — that
    # narrower residual is tracked as a follow-up (persist/memoize the nudge).
    render_idempotency_discriminator: str = ""
    # ``stage_name -> StepLedger`` for a DB-backed run; wired by ``RunDispatcher``.
    # ``None`` everywhere else (unit tests, dry runs, any hand-built context), where
    # :meth:`step_ledger` hands back a ``NullStepLedger``. Read it through
    # :meth:`step_ledger`, never directly. Excluded from eq/repr: it is plumbing,
    # not run state, and it survives ``dataclasses.replace(ctx, ...)`` unchanged —
    # correct, because every replace site keeps ``run_id``.
    step_ledger_factory: Callable[[str], StepLedger] | None = field(
        default=None, compare=False, repr=False
    )

    def step_ledger(self, stage_name: str) -> StepLedger:
        """The step ledger for ``stage_name`` of this run — never ``None``.

        ``await ctx.step_ledger(self.name).once(step_key, input_hash, fn)`` reuses a
        sub-step this stage already finished before a pod death instead of repeating
        the paid call. See ``src/services/stages/shared/step_ledger.py`` for the
        contract and the caller rules.
        """
        if self.step_ledger_factory is not None:
            return self.step_ledger_factory(stage_name)
        # Imported here, not at module top: this module is imported by nearly every
        # stage, and the ledger pulls in the database layer.
        from src.services.stages.shared.step_ledger import NullStepLedger

        return NullStepLedger(stage_name)

    def prior_output(self, stage_name: str) -> dict[str, Any] | None:
        return self.prior_run_outputs.get(stage_name)

    def annotations_for(self, stage_name: str) -> list[dict[str, Any]]:
        return list(self.open_annotations_by_stage.get(stage_name, []))
