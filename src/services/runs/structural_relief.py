"""Post-completion structural-relief regenerate hook (PR-C auto-convergence).

When a ``demo_video`` run finishes with an over-budget timeline that provably CANNOT
converge in place — every over-budget beat is footage-pinned / claim-bearing / locked, so
no in-stage trim/cut/merge can shorten it — ``reconcile_timeline`` classifies the run
``convergence_outcome="cant_converge"`` (orch #671). This hook, called from the run
dispatcher's completion path AFTER the parent run is marked terminal, spawns a bounded child
regenerate run from ``demo_video_gen`` with a smaller ``target_beats`` so gen emits a SHORTER
spine that can meet the budget. It is:

* **Bounded** — the guard (:mod:`structural_relief_policy`) stops at ``max_passes`` (default 2,
  ≤3 paid renders total) and the moment the overrun stops improving.
* **No double-pay** — the child is a fresh ``run_id`` regenerate FROM ``demo_video_gen``
  (`from_stage`), so every stage topologically before it (scrape / source_ingest) is
  inherited as a completed output and never re-billed; only gen→record→compile re-runs.
* **Idempotent under replicas:2 + two completion gates** — the child carries
  ``structural_relief_pass``; the ``uq_runs_parent_relief_pass`` unique partial index makes a
  duplicate spawn of the same ``(parent, pass)`` fail the INSERT, which this hook swallows.
* **Never able to break run completion** — the whole body is wrapped so ANY error (including
  a failed spawn) is logged and turned into a no-op; the parent run stays terminal regardless.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from src.config import get_settings
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_run_tenant
from src.models import beat_fingerprint
from src.services.providers.resolver import PROVIDER_SLOTS_KEY
from src.services.runs.structural_relief_policy import (
    ReliefShape,
    decide_relief_shape,
    decide_structural_relief,
    relief_mode_from_snapshot,
)
from src.services.stages.shared.duration_conformance import (
    DELIVERED_BASIS_PLAYABLE,
)
from src.services.stages.shared.duration_conformance import (
    # "finite float, or None" — the ONE coercion this repo trusts for a duration read off
    # unvalidated config; shared with the stage boundary rather than restated here.
    _usable as finite_or_none,
)
from src.services.stages.shared.duration_plan import MEASURED_SECONDS_PER_BEAT_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.config import PipelineSettings
    from src.database.models import StageOutput
    from src.services.run_engine import RunEngine

log = logging.getLogger(__name__)

_FROM_STAGE = "demo_video_gen"
_RECONCILE_TIMELINE_STAGE = "reconcile_timeline"

#: Pseudo-stage name the paid-spawn decision is persisted under, as a ``stage_outputs``
#: row on the run that was EVALUATED (the prospective parent).
#:
#: WHY ``stage_outputs`` AND NOT ``runs.config_snapshot``. The snapshot is the obvious
#: candidate — it already houses the ``structural_relief`` block — and it is WRONG here,
#: for a measured reason: :func:`src.services.runs.run_setup.order_fingerprint` hashes
#: EVERY top-level snapshot key except the four in ``RUN_SCOPED_SNAPSHOT_KEYS``, and
#: ``api.routes.runs.lifecycle`` compares that hash against the parent's PERSISTED
#: snapshot to decide whether a regenerate may resume (``!=`` ⇒ ``return None`` ⇒ the
#: resume is refused). Writing an audit key onto a completed run's snapshot would
#: therefore silently un-resumable that run — a behaviour change, on the money path, of
#: exactly the shape the ``max_seconds`` equality trap has already caused once. There is
#: no run-scoped events table (every audit model in this repo is tenant/request-scoped
#: with a CHECK-constrained vocabulary and no ``run_id``), so a dedicated table would
#: need a migration. A ``stage_outputs`` row needs none.
#:
#: WHY IT IS ARITHMETICALLY INERT. The name is deliberately NOT a blueprint stage, and
#: every consumer of the row set filters by the blueprint's own topology before deciding
#: anything:
#:
#: * ``run_finalizer.decide_run_terminal_state`` asks ``topo <= terminal_stage_names``
#:   (a SUBSET test — an extra name cannot break it), filters its FAILED branch to
#:   ``stage_name in topo``, and intersects only ``pause_set & completed``.
#: * ``run_setup.reconcile_blueprint_stage_rows`` is INSERT-ONLY and documents that
#:   tolerating rows for stages the blueprint does not declare is deliberate; its
#:   ``missing`` list is derived from ``topological_order()``, so this name cannot
#:   suppress a seed.
#: * ``lifecycle``'s checkpoint strip and resume-point scan both re-filter through
#:   ``body.pause_after_stages`` / ``topological_order()``.
#: * ``RunContext.prior_run_outputs`` does carry it (status ``completed`` + output), and
#:   the one blueprint-AGNOSTIC scanner over those values — ``sound_design.
#:   _any_clip_loudness_unproven`` — reads only a top-level ``degradations`` list. The
#:   payload is nested under a single key precisely so no top-level key of a stage
#:   vocabulary is ever shadowed.
#:
#: Written TERMINAL (``completed``), never left ``pending``: a non-terminal row under a
#: terminal run is what the stranded-stage sweep exists to reap, so a ``pending`` audit
#: row would manufacture exactly the defect class that sweep hunts.
_DECISION_STAGE = "structural_relief_decision"

#: Bumped when the payload shape changes, so an auditor reading a mixed-age population
#: can tell a field that was never written from one that was written empty.
#:
#: v2 (2026-09-16) — ``inputs.shape`` gained ``status_quo_seconds``: what the parent
#: ACTUALLY delivered, the figure the ``STOP_ALREADY_CONFORMING`` guard turns on. A v1
#: row predates that guard entirely; a v2 row carrying ``null`` there means the guard
#: was reachable and the delivery was UNMEASURED. Without the bump those two are
#: indistinguishable on the record.
_DECISION_SCHEMA_VERSION = 2


def _decision_inputs(
    *,
    run: Mapping[str, Any],
    pass_number: int,
    prior_overrun: float | None,
    overrun: float,
    beat_count: int,
    settings: PipelineSettings,
    shape: ReliefShape | None = None,
) -> dict[str, object]:
    """The figures the paid-spawn decision actually turned on, as JSON.

    Only inputs that a gate above reads — nothing derived for display. ``None`` means
    "this gate was never reached", which is why ``shape`` is optional: a run refused by
    ``decide_structural_relief`` never computes a shape, and recording zeros there would
    assert arithmetic that was never performed.
    """
    inputs: dict[str, object] = {
        "convergence_outcome": str(run.get("convergence_outcome") or ""),
        "structural_relief_requested": bool(run.get("structural_relief_requested")),
        "overrun_seconds": overrun,
        "prior_overrun_seconds": prior_overrun,
        "target_seconds": finite_or_none(run.get("target_seconds")),
        # target_seconds is the BODY; the delivered film also carries these card seconds.
        "card_seconds": finite_or_none(run.get("card_seconds")),
        "pass_number": pass_number,
        "max_passes": settings.structural_relief_max_passes,
        "beat_count": beat_count,
        "floor_ratio": settings.demo_timeline_underfull_fail_ratio,
        "infeasible_beat_ids": [str(x) for x in (run.get("infeasible_beat_ids") or [])],
    }
    if shape is not None:
        inputs["shape"] = {
            "verdict": shape.verdict,
            "target_beats": shape.target_beats,
            "shrink_target_beats": shape.shrink_target_beats,
            "resized_for_floor": shape.resized_for_floor,
            "min_beats_for_floor": shape.min_beats_for_floor,
            "measured_seconds_per_beat": shape.measured_seconds_per_beat,
            "seconds_floor": shape.seconds_floor,
            "stop_detail": shape.stop_detail,
            # What the parent ACTUALLY delivered (schema v2), the figure the
            # already-conforming guard turns on. ``None`` = UNMEASURED, which is NOT the
            # same fact as "measured and out of band" — and it is recorded on every
            # verdict, including the ones that never read it, so a future session can
            # re-measure this population rather than re-derive it. A decision nobody can
            # audit later is one nobody can check, and the spend is append-only.
            "status_quo_seconds": shape.status_quo_seconds,
            # Which measure SIZED the child ("delivered" vs "footage") and the figures
            # it was sized from, in that basis's units — so an auditor can tell a cut
            # sized off the narration-tracking delivery from one sized off the footage plan.
            "sizing_basis": shape.sizing_basis,
            "sizing_overrun_seconds": shape.sizing_overrun_seconds,
            "sizing_seconds_per_beat": shape.sizing_seconds_per_beat,
        }
    return inputs


async def _record_relief_decision(
    *,
    run_engine: RunEngine,
    parent_run_id: UUID,
    deliverable_id: UUID,
    config_snapshot: Mapping[str, Any],
    outcome: str,
    refusal: str | None,
    inputs: Mapping[str, Any],
    child_run_id: UUID | None = None,
    child_pass: int | None = None,
) -> None:
    """Persist WHY this evaluation spawned (or refused to spawn) a paid render.

    RECORD-ONLY, and structurally so: this coroutine returns ``None``, no caller branches
    on it, and nothing in this module or any other reads
    ``stage_outputs['structural_relief_decision']`` back. It exists because the decision
    was previously unauditable — a prior session escalated a misspend concern to the owner
    on a premise it then had to refute, purely because the spend left no trace.

    ⛔ IT HAS ITS OWN ``try``, and that is load-bearing rather than defensive habit. The
    caller's body is wrapped in a blanket ``except Exception`` that turns any error into
    ``return None`` — i.e. into NOT SPAWNING. An audit write that raised inside that
    wrapper would therefore SUPPRESS a paid render the decision had already approved,
    making the recorder a silent decision-maker. Swallowing here keeps recording strictly
    downstream of deciding: the worst an unavailable database can do is lose the record.

    The session is the engine's own factory rather than the dispatcher's: the caller
    committed and moved on before this hook ran, and the hook is deliberately not handed
    that session. ``bind_run_tenant`` derives the tenant from the run row itself, so a
    refusal whose snapshot carries NO tenant/subject is still recordable — that refusal
    would otherwise be the one case that could never be audited.
    """
    try:
        sessions = getattr(run_engine, "_sessions", None)
        if sessions is None:
            return
        payload = {
            _DECISION_STAGE: {
                "schema_version": _DECISION_SCHEMA_VERSION,
                "outcome": outcome,
                "refusal": refusal,
                "autoconverge_enabled": True,
                "spawned_child_run_id": str(child_run_id) if child_run_id else None,
                "spawned_pass": child_pass,
                "inputs": dict(inputs),
                "identity": {
                    "evaluated_run_id": str(parent_run_id),
                    "deliverable_id": str(deliverable_id),
                    "tenant_id": _content_str(config_snapshot.get("tenant_id")) or None,
                    "subject_id": _content_str(config_snapshot.get("subject_id")) or None,
                },
            }
        }
        async with sessions() as session:
            await bind_run_tenant(session, parent_run_id)
            repo = StageOutputRepository(session)
            await repo.upsert_pending(parent_run_id, _DECISION_STAGE)
            # Idempotent under replicas:2 and the reconciler backstop: the upsert is
            # ON CONFLICT DO NOTHING and this write is guarded to a NON-TERMINAL row, so
            # the second evaluator's write is a no-op and the first record stands rather
            # than being overwritten by a racing duplicate.
            await repo.mark_completed(parent_run_id, _DECISION_STAGE, payload)
            await session.commit()
    except Exception:
        log.warning(
            "structural relief: could not persist the %s decision record for parent run %s "
            "(ignored — the spawn decision itself is unaffected)",
            outcome,
            parent_run_id,
            exc_info=True,
        )


def _find_reconcile_output(stage_rows: Sequence[StageOutput]) -> Mapping[str, Any]:
    """The ``reconcile_timeline`` stage row's whole output, or ``{}`` when the run never
    ran the relief loop. It carries BOTH signals the spawn needs, already joined: the
    ``timeline_manifest`` (whose ``infeasible_beat_ids`` name the culprits) and the
    post-prepass ``lines`` spine those ids index into — the only place the parent's beat
    OBJECTS (text / scenario_name / gherkin) are in scope on this code path. The manifest's
    own ``beats[]`` rows carry ids/seconds only, never content.
    """
    for row in stage_rows:
        if getattr(row, "stage_name", None) != _RECONCILE_TIMELINE_STAGE:
            continue
        output = getattr(row, "output", None)
        return output if isinstance(output, dict) else {}
    return {}


def _find_delivered_seconds(stage_rows: Sequence[StageOutput]) -> float | None:
    """What this run ACTUALLY delivered, off whichever stage recorded a conformance.

    It searches the rows for a ``duration_conformance`` object rather than naming a
    stage. ``duration_conformance`` is the shared ``DurationConformance`` TypedDict (a
    plain dict on the row's JSONB output), written by
    ``shared/duration_conformance.observe_duration_conformance``.

    ⚠️ THAT IS NOT THE SAME AS "works for every blueprint", and this docstring used to
    claim it was by listing ``lesson_compile`` and ``primitives/compile_stage``. The
    caller returns long before reaching here unless the run has a ``reconcile_timeline``
    stage carrying a ``timeline_manifest``, and only TWO blueprints declare that stage:
    ``demo_video`` and ``showcase`` (``git grep -l reconcile_timeline src/services/
    blueprints/``). ``lesson`` and the ``primitives`` compile stages do stamp a
    conformance, but their runs never reach this function. The reachable set is those
    two, not the fleet. What the stage-agnostic scan actually buys
    is that a RENAME of the mastering stage inside one of them cannot silently turn this
    guard into a no-op — worth having, but a smaller claim than the one it replaced.

    ``None`` when nothing usable is found — no conformance row, or a non-finite figure.
    The caller must read that as UNMEASURED and make no judgement on it; it is the value
    that restores ``decide_relief_shape``'s pre-existing behaviour exactly.

    PREFERS A MEASURED BASIS. ``delivered_basis == DELIVERED_BASIS_PLAYABLE`` is the
    length a player reports; ``"declared"`` is the compiler's nominal timeline, i.e. the
    plan grading itself — comparing a paid child against the PLAN's own claim rather than
    against a measurement is how a truncated master reads as conforming. A declared
    figure is still returned when it is all there is (it is better than no status quo at
    all), but never in preference to a measured one.

    ⛔ THE TIE-BREAK IS NOT ROW ORDER. ``StageOutputRepository.list_for_run`` documents
    that its ordering must never be read as "latest stage", so a last-wins sweep would be
    inferring recency from position — the exact inference that repository forbids. Within
    one basis the LONGEST delivered length wins instead: these rows are all measurements
    of the same deliverable, and a short one is the signature of a truncated or partial
    master (see ``duration_conformance``'s own "container duration is the MAX over
    streams" reasoning). Taking the max makes the answer a property of the VALUES, so it
    is identical whatever order the rows arrive in.
    """
    measured: float | None = None
    declared: float | None = None
    for row in stage_rows:
        output = getattr(row, "output", None)
        if not isinstance(output, dict):
            continue
        conformance = output.get("duration_conformance")
        if not isinstance(conformance, dict):
            continue
        value = finite_or_none(conformance.get("delivered_seconds"))
        if value is None or value <= 0.0:
            continue
        if conformance.get("delivered_basis") == DELIVERED_BASIS_PLAYABLE:
            measured = value if measured is None else max(measured, value)
        else:
            declared = value if declared is None else max(declared, value)
    return measured if measured is not None else declared


def _find_timeline_manifest(reconcile_output: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``timeline_manifest`` dict off the reconcile_timeline output, else None."""
    manifest = reconcile_output.get("timeline_manifest")
    return manifest if isinstance(manifest, dict) else None


def _content_str(value: object) -> str:
    """A content-bearing field read off a persisted (JSON) beat, or ``""``."""
    return value if isinstance(value, str) else ""


def _culprit_fingerprints(
    reconcile_output: Mapping[str, Any], over_budget_ids: Sequence[str]
) -> list[str]:
    """Resolve the flagged POSITIONAL beat ids to CONTENT fingerprints, in spine order.

    ``infeasible_beat_ids`` are ``clip-<n>`` ordinals — the scenario's document-order tag
    in the parent's bdd suite. The relief child RE-RUNS ``demo_video_gen``, regenerating
    that suite with a different scenario count/order, so those ordinals name different
    content in the child (and a stale one can coincidentally match an unrelated beat and
    strip its pin/lock protection). Joining them back to the parent's own beats here —
    where those beat objects actually exist — turns the hint into something regeneration
    cannot invalidate.

    Deterministic: spine order is preserved and duplicates are dropped in first-seen
    order (no set iteration). Beats with no content-bearing field fingerprint to ``""``
    and are skipped — an empty fingerprint must never match anything.
    """
    if not over_budget_ids:
        return []
    raw_lines = reconcile_output.get("lines")
    if not isinstance(raw_lines, list):
        return []
    wanted = {str(x) for x in over_budget_ids}
    seen: set[str] = set()
    fingerprints: list[str] = []
    for raw in raw_lines:
        if not isinstance(raw, dict) or str(raw.get("id") or "") not in wanted:
            continue
        fingerprint = beat_fingerprint(
            scenario_name=_content_str(raw.get("scenario_name")),
            gherkin=_content_str(raw.get("gherkin")),
            text=_content_str(raw.get("text")),
        )
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            fingerprints.append(fingerprint)
    return fingerprints


def _shape_stop_reason(shape: ReliefShape, *, floor_ratio: float) -> str | None:
    """Why this manifest shape must NOT be spawned, or ``None`` to proceed.

    ⛔ ``None`` MEANS SPAWN — the caller bills a render on it. Every non-``SPAWN``
    verdict must therefore yield a non-``None`` string, including one this function has
    never heard of: a new ``ReliefShapeVerdict`` literal that fell through the matches
    below used to FAIL OPEN and bill exactly the render the new guard was added to
    refuse (which is why ``structural_relief_policy._STOP_OUT_OF_BAND`` reuses
    ``STOP_UNDERFULL`` rather than naming itself). The unknown-verdict line at the
    bottom closes that; only ``"SPAWN"`` returns ``None``, and it returns it first.

    WHAT IS PRINTED, and why it is not this function's own sentence any more.
    :attr:`ReliefShape.stop_detail` is written by the policy module at the moment of the
    refusal, from the candidate list the refusal was decided from — it names the counts
    that were enumerated and the two that bracket the band. This function's per-verdict
    wording is a re-derivation from a handful of fields, and on the out-of-band refusal
    it re-derived WRONG: for a 30.00s order, 5 beats, 22.00s measured/beat it printed
    "needs >= 2 beat(s) ... but must shrink below 5" (satisfied by 2, 3 and 4 — arithmetic
    that refuses nothing) and called the miss "underfull" when 2 beats = 44.00s is ABOVE
    the band. Meanwhile the accurate sentence sat on the shape with ZERO consumers. So
    the recorded detail is PREFERRED, and the wording below is the fallback for a shape
    that recorded none.

    All of these protect the same thing — a PAID render — and billing here is
    append-only with no refund path, so each is a refusal to start work that provably
    cannot succeed, never a retry:

    * ``STOP_CANNOT_SHRINK`` — no overrun to relieve, or the lever cannot shrink the
      spine further. A child would re-render an identical spine and burn a pass.
    * ``STOP_UNCONVERGEABLE`` — even a perfectly-chosen ``target_beats``-beat spine
      costs more than the target, so the child cannot converge at ANY choice of beats.
    * ``STOP_ALREADY_CONFORMING`` — the parent's OWN delivery is already inside the
      +/-10% band. A child can at best tie it, and one measured child came back worse
      (+0.04% parent → +5.04% child). This is the only verdict here that refuses a spawn
      for which a legal in-band count DOES exist, which is precisely why it is asked
      first: every other gate reasons about the child alone.
    * ``STOP_UNDERFULL`` — no legal count satisfies both the shrink requirement and the
      duration bounds. Its arithmetic is direction-dependent (under the 50% floor, under
      the band, or over it), which is precisely why it is not restated here: with no
      ``stop_detail`` the fallback reports the verdict and the figures it holds, and
      claims no arithmetic it did not perform.

    Note the asymmetry, and that it is deliberate: where a legal count DOES exist the
    shape re-sizes to it (``resized_for_floor``) instead of stopping. Refusing is the
    last resort, because a refused run delivers nothing.
    """
    if shape.verdict == "SPAWN":
        return None
    if shape.stop_detail:
        # The sentence proved at the point of refusal, by the code that refused.
        return shape.stop_detail
    fallback: dict[str, str] = {
        # Reached only by a shape that recorded NO ``stop_detail`` — the policy module
        # always writes one for this verdict, so in practice this is the belt for a
        # hand-built or future shape, not the sentence production prints.
        "STOP_ALREADY_CONFORMING": (
            f"the parent already delivered within the conformance band of its "
            f"{shape.target_seconds:.2f}s order — refusing to bill a relief render "
            f"against a delivery that already conforms"
        ),
        "STOP_CANNOT_SHRINK": (
            f"cannot shrink (beats={shape.beat_count}, target_beats={shape.target_beats}, "
            f"overrun={shape.overrun_seconds:.2f}s)"
        ),
        "STOP_UNCONVERGEABLE": (
            f"cannot converge at any {shape.target_beats}-beat spine "
            f"(floor={shape.seconds_floor:.2f}s > target={shape.target_seconds:.2f}s, "
            f"beats={shape.beat_count}, overrun={shape.overrun_seconds:.2f}s) — refusing "
            f"to bill an unconvergeable render"
            if shape.seconds_floor is not None
            else f"cannot converge at any {shape.target_beats}-beat spine "
            f"(beats={shape.beat_count}, overrun={shape.overrun_seconds:.2f}s) — refusing "
            f"to bill an unconvergeable render"
        ),
    }
    return fallback.get(shape.verdict) or (
        # NOT a guess at the arithmetic: the shape recorded no detail, so all that is
        # established is the verdict and the numbers it carries. Non-None is the point —
        # a verdict this function cannot word is still a REFUSAL, never a spawn.
        f"refusing to bill a relief render: shape verdict {shape.verdict} with no "
        f"recorded detail (beats={shape.beat_count}, target_beats={shape.target_beats}, "
        f"min_beats_for_floor={shape.min_beats_for_floor}, "
        f"measured/beat={shape.measured_seconds_per_beat or 0.0:.2f}s, "
        f"target={shape.target_seconds:.2f}s, overrun={shape.overrun_seconds:.2f}s, "
        f"floor_ratio={floor_ratio:.0%})"
    )


def _spawn_numbers_are_finite(
    run: Mapping[str, Any], prior_overrun: object, parent_run_id: UUID
) -> bool:
    """Is every figure this spawn decision reads an actual number? Logs the first that is not.

    ``True`` (the healthy answer) means every PRESENT value is finite. An absent value is
    not a fault — these keys are genuinely optional and the caller's existing
    ``or 0.0`` / ``None`` handling owns that case.

    The log line matters as much as the verdict: this function runs on the parent's
    completion path, inside a ``try`` that swallows everything, so a silent ``None`` is
    indistinguishable from a healthy "nothing to do".
    """
    for label, raw_value in (
        ("structural_relief.prior_overrun", prior_overrun),
        ("run.overrun_seconds", run.get("overrun_seconds")),
        ("run.target_seconds", run.get("target_seconds")),
    ):
        if raw_value is not None and finite_or_none(raw_value) is None:
            log.warning(
                "structural relief: parent run %s has a non-finite %s (%r) — refusing "
                "to spawn a paid child render on it",
                parent_run_id,
                label,
                raw_value,
            )
            return False
    return True


def _relief_opted_in(config_snapshot: Mapping[str, Any], parent_run_id: UUID) -> bool:
    """OPT-IN GATE: did THIS run ask for structural relief (``relief.mode == "auto"``)?

    "A manifest is present" does NOT mean "relief ran": reconcile also builds one on a
    relief-OFF run purely to carry footage out-points to the compile. The global
    kill-switch is ON in prod, so without this a relief-off run whose out-points-only
    manifest classified ``cant_converge`` bought a PAID child render nobody opted into
    (PR #1078). Read through the SAME helper reconcile arms its loop on. Verdict:
    STOP_RELIEF_OFF — logged only, NO audit row: a relief-off run is not a relief
    candidate, and the record starts only once a run is one (every trimmed relief-off
    run would otherwise write a row).
    """
    relief_mode = relief_mode_from_snapshot(config_snapshot)
    if relief_mode == "auto":
        return True
    log.info(
        "structural relief: parent run %s has direction.timing.relief.mode=%r "
        "(not 'auto') — STOP_RELIEF_OFF, refusing to spawn a paid child render",
        parent_run_id,
        relief_mode,
    )
    return False


async def maybe_spawn_structural_relief_run(
    *,
    run_engine: RunEngine,
    parent_run_id: UUID,
    deliverable_id: UUID,
    config_snapshot: Mapping[str, Any],
    stage_rows: Sequence[StageOutput],
) -> UUID | None:
    """Spawn a bounded structural-relief regenerate child if the just-completed run asked
    for one. Returns the child run id when spawned, else None. NEVER raises — any failure is
    logged and swallowed so the caller's completion path is unaffected.
    """
    try:
        settings = get_settings().pipeline
        if not settings.structural_relief_autoconverge_enabled:
            return None  # KILL-SWITCH off (default) — never spawn a paid child render
        reconcile_output = _find_reconcile_output(stage_rows)
        manifest = _find_timeline_manifest(reconcile_output)
        # Not a relief run: no reconcile_timeline manifest, OR a manifest that only
        # carries footage out-points on a run that never opted into relief.
        if not manifest or not _relief_opted_in(config_snapshot, parent_run_id):
            return None
        relief_cfg = config_snapshot.get("structural_relief") or {}
        prior_raw = relief_cfg.get("prior_overrun")
        run = manifest.get("run")
        # BOUNDARY, folded into the existing "is this manifest usable at all" guard.
        # Every number read below originates in the unvalidated ``config_overrides`` dict
        # (``json.loads("1e400")`` is a legal JSON number and parses to ``inf``).
        # ``float()`` accepted a non-finite value and the ``<= 0`` progress comparisons
        # downstream are all False against ``nan``, so a corrupt figure reached the
        # PAID-spawn decision without tripping anything. The refusal is this function's
        # existing one — ``return None``, no spawn: it withholds a paid render rather
        # than claiming one is impossible, and substitutes no duration of its own.
        if not isinstance(run, dict) or not _spawn_numbers_are_finite(
            run, prior_raw, parent_run_id
        ):
            # Figures are recorded as REPR, not as numbers: the whole reason this branch
            # fired is that one of them may be ``inf``/``nan``, which JSONB cannot hold —
            # serialising it raw would raise, and the record of the refusal would be the
            # one thing lost to the refusal's own cause.
            await _record_relief_decision(
                run_engine=run_engine,
                parent_run_id=parent_run_id,
                deliverable_id=deliverable_id,
                config_snapshot=config_snapshot,
                outcome="declined",
                refusal="non_finite_inputs",
                inputs={
                    "convergence_outcome": (
                        str(run.get("convergence_outcome") or "") if isinstance(run, dict) else None
                    ),
                    "manifest_run_is_mapping": isinstance(run, dict),
                    "pass_number": int(relief_cfg.get("pass") or 0),
                    "overrun_seconds_repr": repr(
                        run.get("overrun_seconds") if isinstance(run, dict) else None
                    ),
                    "target_seconds_repr": repr(
                        run.get("target_seconds") if isinstance(run, dict) else None
                    ),
                    "prior_overrun_repr": repr(prior_raw),
                },
            )
            return None
        beats = manifest.get("beats") or []
        pass_number = int(relief_cfg.get("pass") or 0)
        prior_overrun = float(prior_raw) if prior_raw is not None else None
        overrun = float(run.get("overrun_seconds") or 0.0)

        decision = decide_structural_relief(
            convergence_outcome=str(run.get("convergence_outcome") or ""),
            structural_relief_requested=bool(run.get("structural_relief_requested")),
            overrun_seconds=overrun,
            prior_overrun_seconds=prior_overrun,
            pass_number=pass_number,
            max_passes=settings.structural_relief_max_passes,
        )
        if decision != "SPAWN":
            log.info(
                "structural relief: parent run %s decision=%s (pass=%s, overrun=%.2fs) — no spawn",
                parent_run_id,
                decision,
                pass_number,
                overrun,
            )
            await _record_relief_decision(
                run_engine=run_engine,
                parent_run_id=parent_run_id,
                deliverable_id=deliverable_id,
                config_snapshot=config_snapshot,
                outcome="declined",
                # The policy verdict IS the refusal identity — STOP_CONVERGED /
                # STOP_CANT_CONVERGE / STOP_CAP answer three different audit questions
                # ("nothing to fix", "unfixable", "budget exhausted"), so the verdict is
                # recorded verbatim rather than collapsed to a single "declined".
                refusal=decision,
                inputs=_decision_inputs(
                    run=run,
                    pass_number=pass_number,
                    prior_overrun=prior_overrun,
                    overrun=overrun,
                    beat_count=len(beats),
                    settings=settings,
                ),
            )
            return None

        tenant_id = config_snapshot.get("tenant_id")
        subject_id = config_snapshot.get("subject_id")
        if not tenant_id or not subject_id:
            log.warning(
                "structural relief: parent run %s has no tenant/subject in snapshot — skip spawn",
                parent_run_id,
            )
            # Recordable BECAUSE ``bind_run_tenant`` reads the tenant off the run row
            # rather than off this (empty) snapshot. Were the recorder to depend on the
            # snapshot's tenant, this exact refusal — the only one caused by that field
            # being absent — would be the one refusal no auditor could ever see.
            await _record_relief_decision(
                run_engine=run_engine,
                parent_run_id=parent_run_id,
                deliverable_id=deliverable_id,
                config_snapshot=config_snapshot,
                outcome="declined",
                refusal="missing_tenant_or_subject",
                inputs=_decision_inputs(
                    run=run,
                    pass_number=pass_number,
                    prior_overrun=prior_overrun,
                    overrun=overrun,
                    beat_count=len(beats),
                    settings=settings,
                ),
            )
            return None

        next_pass = pass_number + 1
        # SHAPE GUARDS — the same pure decision ``reconcile_timeline`` uses to word its
        # degrade note, so the note's forecast and this spawn cannot disagree. Both guards
        # (cannot-shrink, unconvergeable-floor) live in ``structural_relief_policy``; the
        # behaviour here is unchanged, only its home moved.
        shape = decide_relief_shape(
            run=run,
            beats=beats,
            # The floor the child's own reconcile_timeline will judge it by. Passing it
            # here is what lets the shrink target be checked against the underfull gate
            # BEFORE a render is billed, instead of after one.
            floor_ratio=settings.demo_timeline_underfull_fail_ratio,
            # WHAT THE PARENT ACTUALLY SHIPPED — the status quo a paid child has to beat.
            # Without it the shape can only ask "can the child converge?", never "is the
            # child better than the thing it replaces?", and it got that wrong in both
            # directions: it billed a child on a parent already delivering +0.04% (the
            # child came back at +5.04%), and it refused children on parents shipping
            # +23.88 / +74.58 / +23.23 / +32.34 % where a legal count would have landed
            # within ~7s of the order. ``None`` (unmeasured) restores the old behaviour
            # exactly — no judgement is invented from a missing measurement.
            delivered_seconds=_find_delivered_seconds(stage_rows),
        )
        stop_reason = _shape_stop_reason(
            shape, floor_ratio=settings.demo_timeline_underfull_fail_ratio
        )
        if stop_reason:
            log.info("structural relief: parent run %s — %s; no spawn", parent_run_id, stop_reason)
            await _record_relief_decision(
                run_engine=run_engine,
                parent_run_id=parent_run_id,
                deliverable_id=deliverable_id,
                config_snapshot=config_snapshot,
                outcome="declined",
                # The shape verdict names the GATE; ``stop_reason`` is the sentence the
                # refusal proved (preferring ``shape.stop_detail``, which is written from
                # the enumerated candidate list at the moment of refusal). Both are kept:
                # the verdict is what an auditor groups a population by, the sentence is
                # what makes a single row explicable without re-running the arithmetic.
                refusal=shape.verdict,
                inputs={
                    **_decision_inputs(
                        run=run,
                        pass_number=pass_number,
                        prior_overrun=prior_overrun,
                        overrun=overrun,
                        beat_count=len(beats),
                        settings=settings,
                        shape=shape,
                    ),
                    "stop_reason": stop_reason,
                },
            )
            return None
        target_beats = shape.target_beats
        if shape.resized_for_floor:
            # RE-SIZED, not refused: the shrink target the overrun arithmetic asked for
            # was below the underfull floor, so it was raised to the smallest count that
            # can still reach it. A refused run delivers nothing; a re-sized one may still
            # converge, and if it does not it degrades honestly on the next pass.
            log.info(
                "structural relief: parent run %s target_beats re-sized %s -> %s to keep "
                "the child above the underfull floor (%.2fs measured/beat, %.0f%% of "
                "%.2fs required)",
                parent_run_id,
                shape.shrink_target_beats,
                target_beats,
                shape.measured_seconds_per_beat or 0.0,
                100.0 * settings.demo_timeline_underfull_fail_ratio,
                shape.target_seconds,
            )
        over_budget_ids = [str(x) for x in (run.get("infeasible_beat_ids") or [])]
        child = await run_engine.start_run(
            deliverable_id=deliverable_id,
            tenant_id=UUID(str(tenant_id)),
            subject_id=UUID(str(subject_id)),
            parent_run_id=parent_run_id,
            pause_after_stages=[],
            from_stage=_FROM_STAGE,
            # System-spawned re-render: keep the parent's provider pins.
            inherit_slots=config_snapshot.get(PROVIDER_SLOTS_KEY),
            config_overrides={
                "structural_relief": {
                    "pass": next_pass,
                    "target_beats": target_beats,
                    "prior_overrun": overrun,
                    # reconcile_timeline's infeasible-beat ids, kept for BACKWARDS
                    # COMPATIBILITY: an in-flight child (spawned before fingerprints
                    # existed) still reads them, and on the AUTHORED path (stable ids
                    # rebuilt from the brief) they remain exact. gen consults them only
                    # when no fingerprints ride along — they are positional ordinals and
                    # a regenerated bdd suite renumbers them.
                    "over_budget_ids": over_budget_ids,
                    # The SAME culprits keyed by CONTENT (scenario_name / gherkin / text
                    # digest). This is the signal that survives the child's bdd
                    # regeneration, so the cut lands on the beat that actually caused the
                    # overrun instead of on whatever now sits at that ordinal. Empty when
                    # the parent's spine is unavailable ⇒ gen falls back to the ids.
                    "over_budget_fingerprints": _culprit_fingerprints(
                        reconcile_output, over_budget_ids
                    ),
                    # The parent's MEASURED footage yield per pinned beat, carried
                    # forward so the child's `demo_video_gen` plans its scenario count
                    # from a number the recorder actually produced. This is the single
                    # owner of that value on the child side — `gen` reads it, never
                    # re-derives it, and never substitutes a constant when it is absent.
                    MEASURED_SECONDS_PER_BEAT_KEY: shape.measured_seconds_per_beat,
                }
            },
            structural_relief_pass=next_pass,
        )
        log.info(
            "structural relief: parent run %s spawned child %s (pass %s, target_beats %s, "
            "overrun %.2fs)",
            parent_run_id,
            child.id,
            next_pass,
            target_beats,
            overrun,
        )
        # AFTER the spawn, deliberately. The child run id is the only field that makes
        # the spend joinable to the render it paid for, and it does not exist until
        # ``start_run`` returns — a record written before the spawn could name a pass that
        # was never billed (``start_run`` can still raise, or trip the unique guard).
        await _record_relief_decision(
            run_engine=run_engine,
            parent_run_id=parent_run_id,
            deliverable_id=deliverable_id,
            config_snapshot=config_snapshot,
            outcome="spawned",
            refusal=None,
            child_run_id=child.id,
            child_pass=next_pass,
            inputs={
                **_decision_inputs(
                    run=run,
                    pass_number=pass_number,
                    prior_overrun=prior_overrun,
                    overrun=overrun,
                    beat_count=len(beats),
                    settings=settings,
                    shape=shape,
                ),
                # What the child was actually ORDERED to do — the spend's terms, not just
                # its cause. ``target_beats`` is the lever that was moved.
                "spawned_target_beats": target_beats,
                "spawned_from_stage": _FROM_STAGE,
            },
        )
        return child.id
    except IntegrityError:
        # Another replica / the reconciler backstop already spawned this (parent, pass)
        # child — the unique partial index rejected the duplicate INSERT. Expected; no-op.
        log.info(
            "structural relief: parent run %s child pass already spawned (unique guard) — no-op",
            parent_run_id,
        )
        return None
    except Exception:
        # This hook must NEVER turn a legitimate run completion into a failure.
        log.exception(
            "structural relief: spawn hook failed for parent run %s (ignored — the parent run "
            "stays terminal, completion is unaffected)",
            parent_run_id,
        )
        return None
