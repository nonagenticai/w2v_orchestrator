"""Provisioning audit repository (admin tier Phase C-2).

Adopts fr's provisioning_audit lifecycle: a ``pending`` row is written + committed
on its OWN session BEFORE the Keycloak/licenses side effects, then ``finish`` flips
it to ``success`` or ``partial`` (also its own commit). Using a dedicated session
(not the route's request session) is deliberate — if provisioning crashes
mid-flight the pending row is already durable, so an operator can see that a KC
user / license may have been created without a ``success`` row.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.database.models import ProvisioningAudit


class ProvisioningRepository:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def start(
        self, *, actor_subject_id: UUID, target_email: str, payload: dict[str, Any]
    ) -> UUID:
        """Insert a ``pending`` audit row and commit it on its own session."""
        audit_id = uuid7()
        async with self._session_factory() as session:
            session.add(
                ProvisioningAudit(
                    id=audit_id,
                    actor_subject_id=actor_subject_id,
                    action="provision",
                    target_email=target_email,
                    payload=payload,
                    status="pending",
                )
            )
            await session.commit()
        return audit_id

    async def finish(
        self,
        audit_id: UUID,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        target_subject_id: UUID | None = None,
        tenant_id: UUID | None = None,
    ) -> None:
        """Flip the row to ``success``/``partial`` with the created ids + any error."""
        async with self._session_factory() as session:
            await session.execute(
                update(ProvisioningAudit)
                .where(ProvisioningAudit.id == audit_id)
                .values(
                    status=status,
                    result=result,
                    error=error,
                    target_subject_id=target_subject_id,
                    tenant_id=tenant_id,
                )
            )
            await session.commit()
