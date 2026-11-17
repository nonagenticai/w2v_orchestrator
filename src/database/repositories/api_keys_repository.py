"""Tenant API keys (P2 tenancy, migration 0074).

``create`` mints a ``w2v_sk_<prefix>_<secret>`` key (``src/auth/api_key_format``)
and stores only ``prefix`` + ``sha256(full key)``; the full key is returned ONCE.
``resolve`` turns a presented key into its identity: parse → SHA-256 the full key
→ look it up by ``(prefix, hash)`` → reject revoked / expired.

RLS (0090): the table carries the ``tenant_isolation`` policy, so under enforce
every read/mutation here sees ONLY the tenant bound to ``app.current_tenant`` —
callers bind it (``bind_session_tenant``) before calling. Every tenant-scoped
method ALSO filters on the acting ``tenant_id`` (defence in depth: the policy is
inert for the table owner and under ``TENANT_RLS_MODE=off``). The two lookups
that happen BEFORE a tenant is known go through ``SECURITY DEFINER`` resolvers
(``src.database.rls``): ``resolve`` → ``resolve_api_key`` and ``create``'s
global prefix check → ``api_key_prefix_taken``.

Nothing here commits: the caller's session owns the transaction.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from uuid import UUID, uuid7

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.api_key_format import generate_api_key, hash_api_key, parse_api_key
from src.database.models import TenantApiKey

#: A random 40-bit prefix collides with an existing one with negligible but
#: nonzero probability; the UNIQUE constraint would raise. Re-roll a few times.
_PREFIX_ATTEMPTS = 5

#: Pre-tenant resolvers (SECURITY DEFINER, see ``src.database.rls``). They see
#: every tenant's rows regardless of ``app.current_tenant``.
_RESOLVE_SQL = text(
    "SELECT key_id, tenant_id, scopes, revoked_at, expires_at "
    "FROM resolve_api_key(:prefix, :key_hash)"
)
_PREFIX_TAKEN_SQL = text("SELECT api_key_prefix_taken(:prefix)")


@dataclass(frozen=True, slots=True)
class ResolvedApiKey:
    """What authenticating a key yields — never the hash, name or creator."""

    id: UUID
    tenant_id: UUID
    scopes: list[str]


class ApiKeysRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: UUID,
        name: str,
        created_by: UUID | None,
        expires_at: datetime.datetime | None = None,
        scopes: list[str] | None = None,
        partner_licence_id: UUID | None = None,
    ) -> tuple[TenantApiKey, str]:
        """Mint a key; return ``(row, full_key)``. The full key is never stored —
        the caller must hand it to the user now or it is gone.

        ``partner_licence_id`` marks a key minted through the partner lane (P9)."""
        for _ in range(_PREFIX_ATTEMPTS):
            generated = generate_api_key()
            # Prefixes are UNIQUE across ALL tenants; a plain SELECT is RLS-scoped
            # to the acting tenant under enforce and would miss another tenant's.
            taken = (
                await self.session.execute(_PREFIX_TAKEN_SQL, {"prefix": generated.prefix})
            ).scalar_one()
            if not taken:
                break
        else:  # pragma: no cover - 5 consecutive 40-bit collisions
            raise RuntimeError("could not allocate a unique API key prefix")
        row = TenantApiKey(
            id=uuid7(),
            tenant_id=tenant_id,
            name=name,
            prefix=generated.prefix,
            key_hash=generated.key_hash,
            created_by=created_by,
            expires_at=expires_at,
            scopes=sorted(set(scopes or ())),
            partner_licence_id=partner_licence_id,
        )
        self.session.add(row)
        await self.session.flush()
        return row, generated.full_key

    async def list_for_tenant(
        self, tenant_id: UUID, *, include_revoked: bool = False
    ) -> list[TenantApiKey]:
        stmt = select(TenantApiKey).where(TenantApiKey.tenant_id == tenant_id)
        if not include_revoked:
            stmt = stmt.where(TenantApiKey.revoked_at.is_(None))
        stmt = stmt.order_by(TenantApiKey.created_at.desc(), TenantApiKey.id.desc())
        return list((await self.session.execute(stmt)).scalars().all())

    async def revoke(self, key_id: UUID, tenant_id: UUID) -> bool:
        """Stamp ``revoked_at`` on a live key of the acting tenant. False when the
        key is unknown, another tenant's, or already revoked."""
        result = await self.session.execute(
            update(TenantApiKey)
            .where(
                TenantApiKey.id == key_id,
                TenantApiKey.tenant_id == tenant_id,
                TenantApiKey.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
            .returning(TenantApiKey.id)
        )
        return result.first() is not None

    async def live_by_name(self, tenant_id: UUID, name: str) -> TenantApiKey | None:
        """The tenant's LIVE (unrevoked, unexpired) key called ``name``, if any —
        the partner mint's idempotency key. Newest first when several exist (a
        tenant_admin may have reused a name in the UI)."""
        return (
            (
                await self.session.execute(
                    select(TenantApiKey)
                    .where(
                        TenantApiKey.tenant_id == tenant_id,
                        TenantApiKey.name == name,
                        TenantApiKey.revoked_at.is_(None),
                        or_(
                            TenantApiKey.expires_at.is_(None),
                            TenantApiKey.expires_at > func.now(),
                        ),
                    )
                    .order_by(TenantApiKey.created_at.desc(), TenantApiKey.id.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    async def revoke_partner_key(
        self, key_id: UUID, tenant_id: UUID, partner_licence_id: UUID
    ) -> bool:
        """Revoke a live key of ``tenant_id`` that THIS partner minted. False when
        the key is unknown, another tenant's, a tenant_admin's / another
        partner's, or already revoked — a partner never revokes a human's key."""
        result = await self.session.execute(
            update(TenantApiKey)
            .where(
                TenantApiKey.id == key_id,
                TenantApiKey.tenant_id == tenant_id,
                TenantApiKey.partner_licence_id == partner_licence_id,
                TenantApiKey.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
            .returning(TenantApiKey.id)
        )
        return result.first() is not None

    async def live_scopes(self, key_id: UUID, tenant_id: UUID) -> list[str] | None:
        """Scopes of a LIVE key of ``tenant_id`` (None: unknown, another tenant's,
        revoked or expired — a key revoked mid-request loses its grant here)."""
        row = (
            await self.session.execute(
                select(TenantApiKey.scopes).where(
                    TenantApiKey.id == key_id,
                    TenantApiKey.tenant_id == tenant_id,
                    TenantApiKey.revoked_at.is_(None),
                    or_(
                        TenantApiKey.expires_at.is_(None),
                        TenantApiKey.expires_at > func.now(),
                    ),
                )
            )
        ).first()
        return None if row is None else list(row[0] or [])

    async def resolve(
        self, presented: str, *, now: datetime.datetime | None = None
    ) -> ResolvedApiKey | None:
        """The live key for a presented bearer, else None.

        None for: malformed key, unknown prefix, hash mismatch, revoked, or
        expired. All are indistinguishable to the caller on purpose. Runs with NO
        tenant bound (the key is what names the tenant), so it goes through the
        ``resolve_api_key`` SECURITY DEFINER resolver, which matches on the public
        prefix AND the SHA-256 of the full key: without the secret, no row.
        """
        parsed = parse_api_key(presented)
        if parsed is None:
            return None
        row = (
            await self.session.execute(
                _RESOLVE_SQL, {"prefix": parsed.prefix, "key_hash": hash_api_key(presented)}
            )
        ).first()
        if row is None or row.revoked_at is not None:
            return None
        at = now or datetime.datetime.now(datetime.UTC)
        if row.expires_at is not None and row.expires_at <= at:
            return None
        return ResolvedApiKey(id=row.key_id, tenant_id=row.tenant_id, scopes=list(row.scopes or []))

    async def touch_last_used(self, key_id: UUID) -> None:
        """Bump ``last_used_at`` (best-effort usage signal for the key list).

        RLS-scoped like every other write: bind the key's tenant first, or under
        enforce this silently updates 0 rows."""
        await self.session.execute(
            update(TenantApiKey).where(TenantApiKey.id == key_id).values(last_used_at=func.now())
        )
