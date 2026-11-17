"""Mirror billing-priced cost into the local ``run_usage`` table.

The orchestrator's ``/usage``, ``/admin`` and run-viewer cost reads aggregate
over ``run_usage`` — which was never written, so they showed $0. Rather than
rewrite every RLS-aware aggregate to call billing inline, this reconciler
populates ``run_usage`` from billing's authoritative per-correlation cost
(``correlation_id = run_id``, ``stage`` in metadata). The existing read SQL then
works unchanged and shows real dollars.

Read-through, not a background job: the reads call ``reconcile_*`` for terminal
runs whose cost is not SETTLED yet (``runs.usage_reconciled_at`` unset). Each
call replaces the run's billing-owned rows with billing's current view, so a
partial early mirror is corrected by the next read; once the run is past its
settle window (``usage_settle_seconds``) the call also marks it settled and
later views are pure local reads. Concurrent calls for one run are serialised
by a per-run advisory lock (``RunUsageRepository.try_lock_mirror``) — taken
WITHOUT waiting, and only for the local write window, so a viewer that finds the
tenant-wide sweep holding a run skips it for this load instead of queueing behind
200 of them. Best-effort: if billing is unreachable nothing is written or marked,
and the cost stays unsettled rather than failing the page.

ONE exception is NOT best-effort: billing configured but no credential to call it
(``DownstreamCredentialMissingError`` — CELL_AUTH_MODE=enforce with
DOWNSTREAM_LICENSE_TOKEN unset). That is a deployment fault, not an outage, and
swallowing it as "0 rows" is exactly how a run reads as a confident $0. The
reconciler records it (ERROR log once per process + ``cost_unavailable_total``)
and RE-RAISES; read paths catch it and serve ``cost_status="unavailable"`` with
``cost_unavailable_reason=COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING`` and no
total. A caller that forgets to catch it 500s — loud — never shows $0.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from src.clients.billing_client import BillingCostUnavailableError
from src.clients.downstream_credential import DownstreamCredentialMissingError
from src.database.repositories.run_usage_repository import RunUsageRepository
from src.observability.metrics import cost_unavailable_total

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING = "billing_credential_missing"

#: Default bound on the billing read. A caller may pass its own (the run page
#: does, tighter), but NO caller is unbounded: an unbounded downstream call held
#: inside a page render is how a slow billing turns into a 503.
DEFAULT_BILLING_TIMEOUT_S = 5.0

# ERROR once per process: every unreconciled run read hits this path, so a
# per-call ERROR would flood the log with one fault. The metric counts each one.
_credential_missing_logged = False


def _record_billing_credential_missing(exc: DownstreamCredentialMissingError) -> None:
    global _credential_missing_logged
    cost_unavailable_total.labels(reason=COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING).inc()
    if not _credential_missing_logged:
        _credential_missing_logged = True
        log.error(
            "run cost UNAVAILABLE — billing is configured but has no credential: %s "
            "Run/admin cost reads report cost_status=unavailable until this is fixed "
            "(logged once per process; see w2v_orchestrator_cost_unavailable_total).",
            exc,
        )


# Each run's cost is sliced into one run_usage row per (stage, model, provider).
_GROUP_BY = ["correlation_id", "stage", "model", "provider"]

# Numeric fields every billing group MUST carry. billing declares them
# non-optional (``cost_usd: Decimal = 0``, ``input_tokens: int``,
# ``output_tokens: int``) and coerces a DB-null cost to 0 before responding, so
# an ABSENT key is never "no data" — it means billing renamed the field.
#
# This is a read contract we cannot gate: billing is developed elsewhere. Absent
# the check, ``group.get("cost_usd") or "0"`` turns a rename into a confident
# $0.00 row, the sweep marks the run reconciled, and revenue attribution is
# permanently wrong with no error and no log. So: key ABSENT is a contract break
# (loud, refuse to write); key PRESENT and null/zero is a legitimately unpriced
# event and is still recorded as 0.
_REQUIRED_GROUP_FIELDS = ("input_tokens", "output_tokens", "cost_usd")


#: One local mirror row, the shape ``RunUsageRepository.replace_for_runs`` takes.
_Row = dict[str, Any]


def _rows_by_run(data: Mapping[str, Any]) -> dict[UUID, list[_Row]] | None:
    """Billing's grouped answer → local mirror rows per run; ``None`` when the
    response breaks the read contract (write nothing, mark nothing)."""
    rows_by_run: dict[UUID, list[_Row]] = defaultdict(list)
    for group in data.get("groups", []):
        key = group.get("key", {})
        cid = key.get("correlation_id")
        if not cid:
            continue
        try:
            run_uuid = UUID(cid)
        except ValueError, TypeError:
            continue
        missing = [f for f in _REQUIRED_GROUP_FIELDS if f not in group]
        if missing:
            # Unknown response shape — degrade exactly like a billing outage:
            # write nothing, mark nothing, so the run is retried rather than
            # frozen at a fabricated $0.00.
            log.error(
                "run_usage reconcile aborted — billing read contract broken: "
                "group is missing %s; got keys %s. Nothing written; runs stay "
                "unreconciled until the shape is understood.",
                ", ".join(missing),
                sorted(group),
            )
            return None
        rows_by_run[run_uuid].append(
            {
                "stage_name": key.get("stage") or "unknown",
                "model": key.get("model") or None,
                "provider": key.get("provider") or None,
                # A non-LLM cost event (TTS/render/upload) legitimately
                # carries ``input_tokens``/``output_tokens``/``cost_usd`` as
                # explicit ``null``, so ``.get(k, 0)`` returns None (the key
                # exists) — ``or`` coerces that to the numeric default so the
                # repo's int()/Decimal() never sees None.
                "input_tokens": group.get("input_tokens") or 0,
                "output_tokens": group.get("output_tokens") or 0,
                "cost_usd": group.get("cost_usd") or "0",
            }
        )

    # Billing groups by (stage, model, provider) but the mapping above folds a
    # missing stage to "unknown" and an empty model/provider to None, so two
    # billing groups can land on one local key. Sum them: the mirror holds ONE
    # row per key (unique index uq_run_usage_mirrored_group).
    for run_id, rows in rows_by_run.items():
        rows_by_run[run_id] = _merge_colliding_groups(rows)
    return rows_by_run


def _merge_colliding_groups(rows: list[_Row]) -> list[_Row]:
    """Sum rows that share one local key ``(stage_name, model, provider)``."""
    merged: dict[tuple[str, str | None, str | None], _Row] = {}
    for r in rows:
        key = (r["stage_name"], r["model"], r["provider"])
        if key not in merged:
            merged[key] = dict(r)
            continue
        m = merged[key]
        m["input_tokens"] = int(m["input_tokens"]) + int(r["input_tokens"])
        m["output_tokens"] = int(m["output_tokens"]) + int(r["output_tokens"])
        m["cost_usd"] = str(Decimal(str(m["cost_usd"])) + Decimal(str(r["cost_usd"])))
    return list(merged.values())


class BillingCostReader(Protocol):
    """Structural type for the billing read dependency (BillingServiceClient).

    A Protocol so tests can inject a fake without subclassing the real client
    (whose constructor reads settings).
    """

    async def costs_by_correlation(
        self,
        tenant_id: UUID,
        correlation_ids: list[UUID],
        *,
        group_by: list[str] | None = None,
    ) -> dict[str, Any]: ...


class RunUsageReconciler:
    """Fill ``run_usage`` for runs from billing's priced per-correlation cost."""

    def __init__(self, session: AsyncSession, billing_client: BillingCostReader) -> None:
        self._session = session
        self._repo = RunUsageRepository(session)
        self._billing = billing_client

    async def reconcile(
        self,
        tenant_id: UUID,
        runs: list[tuple[UUID, datetime.datetime]],
        *,
        mark_reconciled: bool = False,
        billing_timeout_s: float | None = DEFAULT_BILLING_TIMEOUT_S,
    ) -> int:
        """Mirror billing cost into run_usage for the given (run_id, created_at) list.

        Returns the number of usage rows written. Commits its own writes so a
        read path can call it and immediately re-read. Best-effort: a billing
        outage logs and returns 0 (no partial state — nothing is deleted unless
        billing answered).

        ``mark_reconciled`` stamps ``runs.usage_reconciled_at`` (settled) so a
        zero-cost run is not re-billed on every load. The sweep and the run-viewer
        read pass it only for runs past their settle window.

        Serialised per run WITHOUT waiting: the mirror lock is taken non-blocking
        AFTER billing answers, so it covers only the local replace + mark +
        commit, and a run another caller already holds is dropped rather than
        waited for (see :meth:`_claim`). Holding the lock across the billing call
        was what made a contended run page wait on the sweep — measured 503 after
        8s versus <0.2s. The cost of the reordering is that two callers can both
        read billing for one run; a duplicate read is cheap, a blocked page is not.

        ``billing_timeout_s`` bounds ONLY the billing read (never the local DB
        writes, so a timeout cannot cancel a half-written replace). A read path
        with its own deadline passes it so a slow billing degrades to "no cost
        this load" instead of eating the caller's whole budget; expiry is
        treated exactly like a billing outage — nothing written, nothing marked.
        """
        if not runs:
            return 0

        # Billing FIRST, no lock held across it — see the docstring.
        try:
            async with asyncio.timeout(billing_timeout_s):
                data = await self._billing.costs_by_correlation(
                    tenant_id, [r for r, _ in runs], group_by=_GROUP_BY
                )
        except BillingCostUnavailableError as exc:
            log.warning("run_usage reconcile skipped — billing unavailable: %s", exc)
            return 0
        except TimeoutError:
            log.warning(
                "run_usage reconcile skipped — billing did not answer within %ss",
                billing_timeout_s,
            )
            return 0
        except DownstreamCredentialMissingError as exc:
            # NOT best-effort (see module docstring): record, then propagate so the
            # read path can report cost_status=unavailable instead of $0.
            # Nothing was written and no lock is held, so nothing to release.
            _record_billing_credential_missing(exc)
            raise

        rows_by_run = _rows_by_run(data)
        if rows_by_run is None:
            return 0

        runs = await self._claim(runs)
        if not runs:
            # Every run is being written by another caller right now, or was
            # settled while we read billing. Locks we did take are held until the
            # transaction ends, so end it.
            await self._session.commit()
            return 0

        # One batched replace (constant statements) instead of N per-run round
        # trips — the tenant-wide sweep reconciles up to 200 runs per call.
        written = await self._repo.replace_for_runs(
            [(run_id, rows_by_run.get(run_id, []), created_at) for run_id, created_at in runs],
            mark_reconciled=mark_reconciled,
        )
        await self._session.commit()
        return written

    async def _claim(
        self, runs: list[tuple[UUID, datetime.datetime]]
    ) -> list[tuple[UUID, datetime.datetime]]:
        """The runs this caller may write now: locked WITHOUT waiting, and unsettled.

        Non-blocking by design. A run whose lock another caller holds is DROPPED,
        not waited for: the tenant-wide sweep holds one transaction across up to
        200 runs, and a run page must never queue behind it. Dropping is safe —
        nothing is written or marked for that run, so its cost stays unsettled and
        the next load retries.

        The unsettled re-check runs UNDER the lock, so a run another caller settled
        while this one was reading billing is dropped rather than overwritten with
        a staler mirror.
        """
        held = await self._repo.try_lock_mirror([r for r, _ in runs])
        if not held:
            return []
        locked = [(r, c) for r, c in runs if r in held]
        unsettled = await self._repo.unsettled_among([r for r, _ in locked])
        return [(r, c) for r, c in locked if r in unsettled]

    async def reconcile_one(
        self,
        tenant_id: UUID,
        run_id: UUID,
        created_at: datetime.datetime,
        *,
        mark_reconciled: bool = False,
        billing_timeout_s: float,
    ) -> int:
        """Reconcile a single run (run-viewer cost tile read-through).

        ``billing_timeout_s`` is REQUIRED: a read-through runs inside a request, so
        its caller states the billing deadline it can afford (no unbounded default).

        ``mark_reconciled`` defaults False so a freshly-terminal run stays live
        (re-queried) until its async cost has certainly flushed. The viewer route
        sets it True only once the run is past the same settle window the
        tenant-wide sweep uses, so a settled zero-cost run is not re-billed on
        every load.
        """
        return await self.reconcile(
            tenant_id,
            [(run_id, created_at)],
            mark_reconciled=mark_reconciled,
            billing_timeout_s=billing_timeout_s,
        )

    async def reconcile_unsettled_for_tenant(self, tenant_id: UUID, *, limit: int = 200) -> int:
        """Reconcile and SETTLE this tenant's terminal, recent runs whose cost is
        not settled yet and that are past their settle window.

        Marks each reconciled run so the next /usage or /admin load does not
        re-query billing for it (the zero-cost runs in particular).
        """
        runs = await self._repo.run_ids_unsettled_usage(tenant_id, limit=limit)
        return await self.reconcile(tenant_id, runs, mark_reconciled=True)
