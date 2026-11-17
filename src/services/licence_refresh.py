"""Tenant licence refresh (P2 tenancy, WP4b; introspection since migration 0089).

A pod-lifetime loop that keeps ``tenants.status`` in step with the licences
service for every tenant it can judge — one with a STORED licence token or a
``licence_ref`` — PER TENANT, through the two routes an off-cluster ingress admits
(``src/services/tenant_licence.py``; no the licences internal routes call remains):

* stored token (unexpired) → ``POST /license/introspect``: active ∧ own tenant ∧
  app w2v → ``active`` and ``licence_ref`` = the token's licence (the token is the
  tenant's own credential, so this holds for partner tenants too); ``active:false``
  or another tenant's / app's token → ``licence_lapsed``;
* no token, or an EXPIRED one → ``GET /licenses/active?tenant_id=T&app_key=w2v``:
  none → ``licence_lapsed``; one or more → ``active``. ``licence_ref`` follows only
  when there is EXACTLY one candidate, never while a token (even an EXPIRED one) is
  stored — a token-owned ref changes only through the token path — and never for a
  PARTNER-TOUCHED tenant (``partner_licence_id`` set): its ref bills the API-key
  lane and only a token or the super_admin bind may set it (P9 review blocker);
* licences unreachable / 5xx / a shape we cannot trust (``unavailable``), a
  definite 4xx from licences or its gateway (``refused`` — a configuration fault),
  or a stored token that will not open (``unreadable`` — needs an operator) →
  that tenant is left EXACTLY as it was (could-not-check is never a verdict) and
  ``licence_refresh_untrusted_total{reason}`` counts it;
* the write is a compare-and-set on the state the verdict was reached on (sealed
  token, ``licence_token_set_at``, ``licence_ref``, no live licence-change lease):
  a rotation / bind that lands while the sweep is in its licences call wins, and
  the stale verdict is dropped (``reason="raced"``).

``suspended`` / ``deleted`` are operator decisions and are never touched. Every
sweep also sets ``tenant_licence_token_expiring`` (stored tokens expiring within
30 days). The loop interval carries ±10% jitter so pods do not sweep in lockstep.
No transaction is held across the licences calls.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.repositories.tenant_licence_repository import (
    StoredLicence,
    TenantLicenceRepository,
)
from src.observability.metrics import (
    licence_refresh_untrusted_total,
    tenant_licence_token_expiring,
)
from src.services.providers.crypto import Keyring, get_keyring
from src.services.tenant_licence import (
    Entitlement,
    LicenceIntrospection,
    LicencesRefusedError,
    LicencesUnavailableError,
    LicenceTokenUnreadableError,
    tenant_entitlement,
)

_log = logging.getLogger("w2v_orchestrator.licence_refresh")

#: At most this many licences calls in flight per sweep.
_CONCURRENCY = 8
_EXPIRY_HORIZON = datetime.timedelta(days=30)


@dataclass
class RefreshResult:
    lapsed: list[UUID] = field(default_factory=list)
    recovered: list[UUID] = field(default_factory=list)
    unchanged: int = 0
    #: Tenants that could not be judged this sweep (left as they were).
    untrusted: int = 0


async def _judge(
    client: LicenceIntrospection, keyring: Keyring, row: StoredLicence
) -> Entitlement | str:
    """The tenant's entitlement, or the untrusted ``reason``."""
    try:
        return await tenant_entitlement(
            client,
            keyring,
            row.tenant_id,
            row.ciphertext,
            token_expires_at=row.token_expires_at,
            site="sweep",
        )
    except LicenceTokenUnreadableError:
        _log.error(
            "licence refresh: tenant %s stored licence token cannot be opened "
            "(rotated/missing PROVIDER_SECRET_KEYS key or corrupt row) — left unchanged",
            row.tenant_id,
        )
        return "unreadable"
    except LicencesRefusedError:
        return "refused"
    except LicencesUnavailableError as exc:
        _log.warning("licence refresh: tenant %s not checked (%s)", row.tenant_id, exc)
        return "unavailable"


def _new_ref(row: StoredLicence, ent: Entitlement) -> UUID | None:
    """The ``licence_ref`` to write, or None to keep it. Only the token path may
    change a token-owned ref: while ANY token is stored — even an EXPIRED one the
    check fell back past — a fallback verdict updates status / checked_at only
    (same rule as ``record_fallback_ref``)."""
    if not ent.entitled or ent.licence_ref is None or ent.licence_ref == row.licence_ref:
        return None
    if ent.source == "fallback" and row.ciphertext is not None:
        return None  # a stored token owns the ref, expired or not
    if ent.source == "fallback" and row.partner_licence_id is not None:
        return None  # never auto-switch a partner tenant's billing ref
    return ent.licence_ref


async def refresh_tenant_licences(
    session: AsyncSession, client: LicenceIntrospection, keyring: Keyring
) -> RefreshResult:
    """One sweep. Commits its own writes; holds no transaction across licences."""
    repo = TenantLicenceRepository(session)
    rows = await repo.sweep_targets()
    tenant_licence_token_expiring.set(await repo.expiring_token_count(_EXPIRY_HORIZON))
    await session.rollback()

    gate = asyncio.Semaphore(_CONCURRENCY)

    async def judge(row: StoredLicence) -> Entitlement | str:
        async with gate:
            return await _judge(client, keyring, row)

    verdicts = await asyncio.gather(*(judge(r) for r in rows))

    result = RefreshResult()
    refused = 0
    for row, verdict in zip(rows, verdicts, strict=True):
        if isinstance(verdict, str):
            licence_refresh_untrusted_total.labels(reason=verdict).inc()
            refused += verdict == "refused"
            result.untrusted += 1
            continue
        if not await repo.record_sweep_verdict(
            row, entitled=verdict.entitled, licence_ref=_new_ref(row, verdict)
        ):
            # The token / ref changed (or a licence-change lease is held) since the
            # read: this verdict is about a state that no longer exists.
            licence_refresh_untrusted_total.labels(reason="raced").inc()
            result.untrusted += 1
            continue
        if not verdict.entitled and row.status != "licence_lapsed":
            result.lapsed.append(row.tenant_id)
        elif verdict.entitled and row.status == "licence_lapsed":
            result.recovered.append(row.tenant_id)
        else:
            result.unchanged += 1
    await session.commit()
    if refused:
        _log.warning(
            "licence refresh: licences (or its gateway) REFUSED %d of %d tenant checks — "
            "a configuration fault of this deployment, not an outage",
            refused,
            len(rows),
        )
    if result.lapsed or result.recovered:
        _log.info(
            "licence refresh: lapsed=%s recovered=%s unchanged=%d untrusted=%d",
            [str(t) for t in result.lapsed],
            [str(t) for t in result.recovered],
            result.unchanged,
            result.untrusted,
        )
    return result


#: ±fraction of the interval added to every sleep so pods do not sweep in lockstep.
_JITTER = 0.1


def jittered(interval_seconds: float) -> float:
    if interval_seconds <= 0:
        return interval_seconds
    return interval_seconds * (1 + random.uniform(-_JITTER, _JITTER))  # noqa: S311 — not crypto


async def run_licence_refresh_loop(
    session_factory: async_sessionmaker[AsyncSession],
    client_factory: Callable[[], LicenceIntrospection],
    interval_seconds: float,
    keyring_factory: Callable[[], Keyring] = get_keyring,
) -> None:
    """Sweep forever every ``interval_seconds`` (±10% jitter). A failed sweep is logged and
    retried next tick — it never kills the loop."""
    while True:
        try:
            async with session_factory() as session:
                await refresh_tenant_licences(session, client_factory(), keyring_factory())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — keep the loop alive
            _log.exception("licence refresh sweep failed")
        await asyncio.sleep(jittered(interval_seconds))


def start_licence_refresh_task(
    session_factory: async_sessionmaker[AsyncSession], interval_seconds: int
) -> asyncio.Task[None] | None:
    """Lifespan hook. None when disabled (``interval_seconds <= 0``)."""
    if interval_seconds <= 0:
        _log.info("licence refresh disabled (interval=%s)", interval_seconds)
        return None
    from src.api.routes._common import get_licenses_client

    return asyncio.create_task(
        run_licence_refresh_loop(session_factory, get_licenses_client, interval_seconds),
        name="licence-refresh",
    )
