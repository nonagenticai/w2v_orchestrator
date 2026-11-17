"""E2E project retention sweep — archives the projects the e2e suites abandon.

Built on the ``audit_prune`` template in this package: a lifespan-started
``asyncio.create_task`` + ``while True`` loop, and a ``_start_*_task`` helper
that returns ``None`` when the feature is not configured.

WHY THIS EXISTS RATHER THAN PER-SPEC TEARDOWN. Measured 2026-09-16: there are
**46 live project create-sites and zero of them delete the project** — 13
Playwright specs under ``w2v_frontend/e2e/tests`` (each inlining its own
``POST /api/projects``; there is no shared factory to add an ``afterAll`` to)
and 33 happy-path scenarios in ``scripts/e2e_scenarios``. The harness does have
a ``cleanup`` contract, but ``_teardown_run`` (``scripts/e2e_validate.py``)
only CANCELS the run — it never touches the project. Two other scenario shapes
(storyboard-merge, panel_image) do ``DELETE /api/projects/{id}`` in a
``finally``; the dominant shape does not. On top of that a ``course`` run
spawns lesson children server-side that no spec ever sees an id for. Patching
46 sites would still miss those children and every ad-hoc run, so the leak is
closed here, once, downstream of all of them.

⛔ WHAT MAKES A MISCONFIGURATION HARMLESS — the three properties to preserve:

1. **Both ids or nothing.** ``_start_e2e_project_prune_task`` returns ``None``
   unless BOTH ``e2e_project_prune_subject_id`` and
   ``e2e_project_prune_tenant_id`` parse as UUIDs. Partial config is the
   dangerous shape: the owner's real projects live in the SAME tenant under a
   different subject, so a tenant-only (or age-only) sweep would archive them.
   There is no wildcard and no "prune everything older than N" mode.
2. **Armed-or-not is logged at startup, exactly once.** A disabled reaper
   nobody notices is its own failure mode — the whole point of this module is
   that projects stop accumulating, and silence cannot distinguish "working"
   from "never started".
3. **A per-sweep cap, with the archived ids logged.** Steady state is ~22
   projects/hour against a cap of 50, so the cap is never reached while the
   scope is right. If it IS reached the scope is wrong, and the cap converts
   that from a mass-archive into a logged anomaly with a bounded blast radius
   that someone can catch before the next tick.

The write itself is a SOFT archive (``status='archived'``), which is what
``DELETE /api/projects/{id}`` does and what removes a project from every
listing. Nothing here hard-deletes, so no cascade and no trigger is involved.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from src.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)

# How many archived ids to name individually in the per-sweep log line. Beyond
# this the line reports the count plus this many as a sample: at the cap of 50 a
# full id list is 1.8KB of log per tick, and the count is the part that tells an
# operator something is wrong.
_LOGGED_ID_SAMPLE = 10

# ⛔ SAFETY INTERLOCK — NOT CONFIGURATION. Do not move this to settings, do not
# make it env-overridable, and do not "clean it up" as a hardcoded value.
#
# Every other guard in this module is aimed at the MECHANISM: the post-condition
# catches a wrong query, the cap catches a runaway batch, the arming check
# catches a partial scope. None of them is aimed at the TARGET. Point
# ``PIPELINE_E2E_PROJECT_PRUNE_SUBJECT_ID`` at a real human's subject and every
# barrier politely agrees — the SELECT matches, the UPDATE matches, and
# ``foreign_owned_ids`` returns empty, because the rows genuinely ARE owned by
# the configured subject. The owner's ~95 terminal projects would archive at 50
# per tick, and `ProjectRepository.recompute_status` returns early on
# ``archived``, so not one of them comes back on its own.
#
# A typo in a ConfigMap is a realistic way to reach that. This list is the only
# thing standing between that typo and a silent, unrecoverable mass-archive of
# someone's real work, so it fails CLOSED: a match refuses to arm at all rather
# than narrowing what gets swept.
#
# Add a subject here whenever a real (human-owned) subject becomes known. Never
# remove one, and never add the e2e subject 8382f0b3-c082-4f97-8d47-120ce5a18e05
# — that is the one this sweep exists to clean up.
_NEVER_PRUNE_SUBJECT_IDS = frozenset(
    {
        # The owner's real subject, in the SAME tenant the e2e suites write to.
        UUID("32969c16-6ecd-49fe-a106-ba13e6640dc8"),
    }
)


def _parse_scope_uuid(raw: str | None, *, field: str) -> UUID | None:
    """Parse a configured scope id, or ``None`` (logging WHY, if it was set).

    A malformed id must disarm the sweep rather than raise: the alternative is a
    typo in a ConfigMap crash-looping the orchestrator. It must also not be
    silent — an unparseable subject that merely disabled the sweep would look
    exactly like a deliberate opt-out.
    """
    if raw is None or not raw.strip():
        return None
    try:
        return UUID(raw.strip())
    except ValueError:
        log.error(
            "e2e project prune: %s=%r is not a UUID; sweep DISABLED",
            field,
            raw,
        )
        return None


async def _sweep_once(
    session_factory: async_sessionmaker[AsyncSession],
    tenant_id: UUID,
    owner_subject_id: UUID,
    min_age_hours: int,
    max_per_sweep: int,
) -> Sequence[UUID]:
    """One pass: archive at most ``max_per_sweep`` stale projects, return their ids.

    Separated from the loop so the scope + cap + logging can be tested without
    driving a ``while True``.
    """
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.session import bind_session_tenant

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=min_age_hours)
    async with session_factory() as session:
        # ``w2v_orch_app`` has no BYPASSRLS, so an unbound background session
        # reads ZERO rows and this sweep would be silently vacuous. Bind the
        # configured tenant explicitly — the same discipline the reconciler and
        # the preview-proxy sweeper use for their own background writes.
        await bind_session_tenant(session, tenant_id)
        repo = ProjectRepository(session)
        archived = await repo.archive_stale_for_subject(
            tenant_id=tenant_id,
            owner_subject_id=owner_subject_id,
            older_than=cutoff,
            limit=max_per_sweep,
        )
        # ⛔ POST-CONDITION, IN THE SAME TRANSACTION, BEFORE COMMIT.
        # The cap bounds the blast per TICK, not in total: 50/tick on an hourly
        # loop still reaches the owner's ~95 real projects within ~2 hours if the
        # scope is ever wrong. So re-read what is about to be committed and prove
        # every row belongs to the configured subject. This is deliberately
        # redundant with the WHERE clause — it is the independent check that a
        # regression in the query (a dropped predicate, a swapped pair of ids)
        # cannot also disable, and rolling back makes such a regression a loud
        # no-op instead of a slow, quiet catastrophe.
        foreign = await repo.foreign_owned_ids(archived, owner_subject_id=owner_subject_id)
        if foreign:
            # ⚠️ This rollback is defence-in-depth, and DELIBERATELY redundant:
            # the `return []` below already skips the commit, and closing the
            # session discards the transaction. Mutation-testing confirms it —
            # deleting this line alone keeps every test green, because the data
            # outcome is doubly guaranteed. It stays because the redundancy is
            # one edit deep: anyone who later moves the commit, or removes the
            # early return, loses the only other thing stopping a foreign-scoped
            # batch from landing. Do not "clean it up" on the grounds that
            # nothing fails without it.
            await session.rollback()
            log.error(
                "e2e project prune: ABORTED — %d of %d projects in this batch are NOT "
                "owned by the configured subject %s. Rolled back, archived NOTHING. "
                "The sweep's scope is wrong; do not re-arm it until the configured "
                "subject/tenant pair is verified. Foreign ids (sample): %s",
                len(foreign),
                len(archived),
                owner_subject_id,
                [str(pid) for pid in foreign[:_LOGGED_ID_SAMPLE]],
            )
            return []
        await session.commit()
    if not archived:
        return archived
    if len(archived) >= max_per_sweep:
        # Reaching the cap is the anomaly signal, not a progress report.
        log.warning(
            "e2e project prune: archived %d projects — THE PER-SWEEP CAP (%d) WAS "
            "REACHED for subject %s in tenant %s. Expected steady state is well "
            "under the cap; verify the configured subject is the e2e one. "
            "Sample: %s",
            len(archived),
            max_per_sweep,
            owner_subject_id,
            tenant_id,
            [str(pid) for pid in archived[:_LOGGED_ID_SAMPLE]],
        )
    elif len(archived) > _LOGGED_ID_SAMPLE:
        log.info(
            "e2e project prune: archived %d projects older than %s for subject %s "
            "(sample of %d): %s",
            len(archived),
            cutoff,
            owner_subject_id,
            _LOGGED_ID_SAMPLE,
            [str(pid) for pid in archived[:_LOGGED_ID_SAMPLE]],
        )
    else:
        log.info(
            "e2e project prune: archived %d projects older than %s for subject %s: %s",
            len(archived),
            cutoff,
            owner_subject_id,
            [str(pid) for pid in archived],
        )
    return archived


async def _run_periodic_e2e_project_prune(
    session_factory: async_sessionmaker[AsyncSession],
    tenant_id: UUID,
    owner_subject_id: UUID,
    min_age_hours: int,
    interval_seconds: int,
    max_per_sweep: int,
) -> None:
    """Sweep every ``interval_seconds`` for the pod's whole lifetime.

    Work FIRST, sleep after — a sleep-first loop never completes a pass in a pod
    that restarts more often than ``interval_seconds``, which is the normal case
    during a rollout, so the sweep would silently never run. (Copied deliberately
    from ``audit_prune``, where the same trap was already paid for.)

    A single bad tick must never kill the loop: transient DB errors are logged
    and the next interval retries.
    """
    while True:
        try:
            await _sweep_once(
                session_factory,
                tenant_id,
                owner_subject_id,
                min_age_hours,
                max_per_sweep,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("e2e project prune tick failed; continuing")
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


def _start_e2e_project_prune_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    """Spawn the e2e retention sweep, or ``None`` when it is not configured.

    Requires BOTH scope ids. Logs whether it armed, either way, so the sweep can
    never be silently inert — see property 2 in the module docstring.
    """
    settings = get_settings()
    pipeline = settings.pipeline
    subject_id = _parse_scope_uuid(
        pipeline.e2e_project_prune_subject_id, field="PIPELINE_E2E_PROJECT_PRUNE_SUBJECT_ID"
    )
    tenant_id = _parse_scope_uuid(
        pipeline.e2e_project_prune_tenant_id, field="PIPELINE_E2E_PROJECT_PRUNE_TENANT_ID"
    )
    if subject_id in _NEVER_PRUNE_SUBJECT_IDS:
        # Fail CLOSED and loudly. This is the misconfiguration no downstream
        # guard can catch, because nothing downstream is wrong — see
        # `_NEVER_PRUNE_SUBJECT_IDS`.
        log.error(
            "e2e project prune: REFUSING TO ARM — the configured subject %s is on the "
            "never-prune list (a known REAL subject, not the e2e one). Every downstream "
            "guard would have accepted this: the rows genuinely belong to the "
            "configured subject, so the scope post-condition passes and the sweep would "
            "archive a real person's projects, unrecoverably. Fix "
            "PIPELINE_E2E_PROJECT_PRUNE_SUBJECT_ID; the e2e subject is the one whose "
            "projects this sweep exists to clean up.",
            subject_id,
        )
        return None
    if subject_id is None or tenant_id is None:
        log.info(
            "e2e project prune: DISABLED (subject configured=%s, tenant configured=%s) — "
            "both PIPELINE_E2E_PROJECT_PRUNE_SUBJECT_ID and "
            "PIPELINE_E2E_PROJECT_PRUNE_TENANT_ID are required; e2e projects will "
            "accumulate in the projects listing",
            subject_id is not None,
            tenant_id is not None,
        )
        return None
    log.info(
        "e2e project prune: ARMED for subject %s in tenant %s — archiving terminal "
        "projects idle >%dh, at most %d per sweep, every %ds",
        subject_id,
        tenant_id,
        pipeline.e2e_project_prune_min_age_hours,
        pipeline.e2e_project_prune_max_per_sweep,
        pipeline.e2e_project_prune_interval_seconds,
    )
    return asyncio.create_task(
        _run_periodic_e2e_project_prune(
            session_factory,
            tenant_id,
            subject_id,
            pipeline.e2e_project_prune_min_age_hours,
            pipeline.e2e_project_prune_interval_seconds,
            pipeline.e2e_project_prune_max_per_sweep,
        ),
        name="e2e-project-prune",
    )
