"""GC of superseded connection secrets.

A superseded version's ``secret_ciphertext`` is purged once no NON-TERMINAL run
(queued, running, **paused** …) still pins it — a paused run resumed next week must
find the exact credential it started with. Only the ciphertext is purged; the row
stays (audit + "pinned version purged" diagnostics).

The whole pass is ONE call to the ``purge_unpinned_provider_secrets`` SECURITY
DEFINER function (``src/database/rls.py``, migration 0079): the pod's runtime role
is RLS-restricted, so it cannot itself see every tenant's runs and versions, and a
single statement leaves no window between reading the pins and purging the rest.
It raises (purges nothing) when a non-terminal run's pins are malformed.

A **grace window** (``PROVIDER_GC_GRACE_SECONDS``, default 1h) keeps every version
superseded less than that long ago: a run that pinned the version a moment before
it was superseded may not have COMMITTED its snapshot yet, so the pin scan cannot
see it. An hour is far longer than any run-start transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: Default grace window — see the module docstring.
DEFAULT_GC_GRACE = timedelta(hours=1)
#: Versions purged per pass at most (the loop catches up on the next tick).
DEFAULT_GC_BATCH = 500


async def purge_unpinned_secrets(
    session: AsyncSession,
    *,
    grace: timedelta = DEFAULT_GC_GRACE,
    now: datetime | None = None,
    limit: int = DEFAULT_GC_BATCH,
) -> int:
    """One GC pass; returns the number of secrets purged. The caller commits."""
    cutoff = (now or datetime.now(UTC)) - grace
    result = await session.execute(
        text("SELECT purge_unpinned_provider_secrets(:cutoff, :limit)"),
        {"cutoff": cutoff, "limit": limit},
    )
    return int(result.scalar_one() or 0)
