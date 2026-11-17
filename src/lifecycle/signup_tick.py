"""Pod-lifetime loop finishing approved early-access signups (every 5 min).

It also runs DUE early-access REMOVALS (``src/services/signup/removal.py``):
same lease, same backoff, same 10-attempt cap.

An APPROVED signup whose provisioning steps did not all land (a licences /
provisioning blip; the set-password email waiting for them) is re-run under the
row's steps lease with a bounded backoff (5 min doubling, capped at 6 h, at most
10 attempts — then the super_admin Retry). Safe with several replicas: every row
is taken with a conditional-UPDATE lease, so two pods never run the same row.
Same shape as ``offboarding_resume``: work first, sleep after; a failed tick is
logged and retried next interval; the steps pass and the removal pass each run
under their OWN try/except (one failing never starves the other). Each tick is inert unless
``W2V_SIGNUP_ENABLED=true`` (read per tick). ``W2V_SIGNUP_TICK_INTERVAL_SECONDS``
(default 300; ``<= 0`` disables the loop — and then the removal start / retry
routes refuse with 503 ``removal_worker_disabled`` instead of queueing silently).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.api.routes._common import (
    get_licence_keyring,
    get_licenses_client,
    get_signup_keycloak_admin,
)
from src.config import current_signup_settings
from src.observability.metrics import db_errors_total
from src.services import tenant_offboarding as offboarding
from src.services.signup import core, removal
from src.services.signup.telegram import log_telegram_status
from src.services.tts_storage import _get_storage

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)


def tick_deps() -> core.SignupDeps:
    return core.SignupDeps(
        licenses=get_licenses_client(),
        identity=get_signup_keycloak_admin(),
        keyring=get_licence_keyring(),
    )


async def removal_deps() -> removal.RemovalDeps:
    return removal.RemovalDeps(
        licenses=get_licenses_client(),
        identity=get_signup_keycloak_admin(),
        storage=offboarding.as_offboarding_storage(await _get_storage()),
    )


async def _run_steps_tick(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Due approved rows' steps. A failure is logged + counted, never raised."""
    try:
        return await core.run_tick(session_factory, tick_deps())
    except asyncio.CancelledError:
        raise
    except Exception:
        db_errors_total.labels(op="signup_tick_steps").inc()
        log.exception("signup tick: the approved-steps pass failed; retrying next tick")
        return 0


async def _run_removal_tick(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Due early-access REMOVALS — INDEPENDENT of the steps pass: a failing
    ``core.run_tick`` never starves a removal (and vice versa). A failure is
    logged + counted (``db_errors_total{op="signup_tick_removals"}``)."""
    try:
        return await removal.run_due_removals(session_factory, await removal_deps())
    except asyncio.CancelledError:
        raise
    except Exception:
        db_errors_total.labels(op="signup_tick_removals").inc()
        log.exception("signup tick: the removal pass failed; retrying next tick")
        return 0


async def signup_tick_once(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Due approved rows' steps, then due early-access REMOVALS (each on its own)."""
    if not current_signup_settings().enabled:
        return 0
    ran = await _run_steps_tick(session_factory)
    removed = await _run_removal_tick(session_factory)
    if ran or removed:
        log.info("signup tick: step runs=%d removal runs=%d", ran, removed)
    return ran + removed


def removal_worker_enabled() -> bool:
    """The signup tick (the only thing that runs a removal) is configured to run."""
    return current_signup_settings().tick_interval_seconds > 0


async def _run_periodic_signup_tick(
    session_factory: async_sessionmaker[AsyncSession], interval_seconds: int
) -> None:
    while True:
        try:
            await signup_tick_once(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("signup tick failed; retrying next tick")
        await asyncio.sleep(interval_seconds)


def start_signup_tick_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    cfg = current_signup_settings()
    log.info("signup: enabled=%s", cfg.enabled)
    log_telegram_status()
    interval = cfg.tick_interval_seconds
    if interval <= 0:
        log.info("signup tick disabled (interval=%s)", interval)
        return None
    return asyncio.create_task(
        _run_periodic_signup_tick(session_factory, interval), name="signup-tick"
    )
