"""Bridge the resolver's protocols (WP2) to WP1's repository and keyring.

The resolver codes against two small protocols (``ConnectionRepository``,
``SecretDecryptor``) so it can be unit-tested with fakes. WP1 ships a richer
repository (``get_current`` → ``(connection, version)``, a tenant-scoped
``get_version(tenant_id, cid, v)``) and a keyring whose ``decrypt`` takes the AAD
parts as keywords. These adapters are the only place the two shapes meet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from src.database.models import TenantConnection, TenantConnectionVersion
from src.database.repositories.connections_repository import ConnectionsRepository
from src.services.providers.byo_posture import ByoPostureReader, get_byo_posture
from src.services.providers.crypto import Keyring
from src.services.providers.settings import ProviderPolicySettings, get_provider_policy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.config import Settings


@dataclass(frozen=True, slots=True)
class ConnectionVersionView:
    """A version joined with its connection's tenant + slot (``ConnectionVersionRecord``)."""

    tenant_id: UUID
    slot: str
    connection_id: UUID
    version: int
    kind: str
    base_url: str | None
    model: str | None
    options: Mapping[str, object] = field(default_factory=dict)
    secret_ciphertext: bytes | None = None
    purged_at: object | None = None

    @classmethod
    def of(cls, conn: TenantConnection, ver: TenantConnectionVersion) -> ConnectionVersionView:
        return cls(
            tenant_id=conn.tenant_id,
            slot=conn.slot,
            connection_id=ver.connection_id,
            version=ver.version,
            kind=ver.kind,
            base_url=ver.base_url,
            model=ver.model,
            options=dict(ver.options or {}),
            secret_ciphertext=(
                bytes(ver.secret_ciphertext) if ver.secret_ciphertext is not None else None
            ),
            purged_at=ver.purged_at,
        )


class DbConnectionRepository:
    """``ConnectionRepository`` over WP1's :class:`ConnectionsRepository`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = ConnectionsRepository(session)

    async def get_active(self, tenant_id: UUID, slot: str) -> ConnectionVersionView | None:
        current = await self._repo.get_current(tenant_id, slot)
        return None if current is None else ConnectionVersionView.of(*current)

    async def get_version(self, connection_id: UUID, version: int) -> ConnectionVersionView | None:
        """Exactly that version, joined with its connection (any status).

        Not tenant-filtered here on purpose: the resolver compares the record's
        ``tenant_id`` with the run's and raises ``SlotUnavailableError`` on a
        mismatch, which is a clearer failure than "no longer exists". RLS is the
        backstop under the restricted runtime role."""
        row = (
            await self._session.execute(
                select(TenantConnection, TenantConnectionVersion)
                .join(
                    TenantConnectionVersion,
                    TenantConnectionVersion.connection_id == TenantConnection.id,
                )
                .where(
                    TenantConnectionVersion.connection_id == connection_id,
                    TenantConnectionVersion.version == version,
                )
            )
        ).first()
        return None if row is None else ConnectionVersionView.of(row[0], row[1])


class KeyringDecryptor:
    """``SecretDecryptor`` over a :class:`Keyring` (AAD already built by the resolver)."""

    def __init__(self, keyring: Keyring) -> None:
        self._keyring = keyring

    def decrypt(self, ciphertext: bytes, aad: bytes) -> str:
        return self._keyring.decrypt_with_aad(ciphertext, aad)


@dataclass(frozen=True, slots=True)
class ResolverDeps:
    repo: DbConnectionRepository
    decryptor: KeyringDecryptor
    policy: ProviderPolicySettings
    posture: ByoPostureReader


def build_resolver_deps(session: AsyncSession, settings: Settings) -> ResolverDeps:
    """Everything ``resolve_slot`` / ``pin_slots`` need, bound to ``session``."""
    keyring = Keyring(settings.provider_secrets.keyring_entries())
    return ResolverDeps(
        repo=DbConnectionRepository(session),
        decryptor=KeyringDecryptor(keyring),
        policy=get_provider_policy(),
        posture=get_byo_posture(),
    )
