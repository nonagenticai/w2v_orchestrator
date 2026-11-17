"""Tenant provider connections (P3 provider slots, migration 0077).

A connection binds one ``(tenant, slot)`` to a provider; every change writes a NEW
immutable :class:`TenantConnectionVersion` and moves ``current_version``. Runs pin
``(connection_id, version)``, so an old version must stay readable (and its secret
decryptable) until no non-terminal run references it — that decision belongs to
the GC (``src/services/providers/gc.py``); this module only offers the candidate
query (:meth:`ConnectionsRepository.purge_candidates`) and the purge itself.

Both tables are under RLS ``tenant_isolation``, and EVERY tenant-scoped statement
here still filters on the acting ``tenant_id`` explicitly — the owner role (tests,
``TENANT_RLS_MODE=off``) bypasses the policy, so the WHERE clause is the guarantee,
the policy the backstop. The cross-tenant scans (:meth:`purge_candidates`,
:meth:`secrets_to_rewrap`) see only the bound tenant under the restricted runtime
role; they are meant for maintenance sessions on the owner connection.

Secret semantics of :meth:`ConnectionsRepository.create_version`:

* ``KEEP_SECRET`` (the default) — carry the previous version's secret forward. The
  ciphertext is bound to its row (AAD = tenant|connection|version), so it is
  decrypted and RE-SEALED for the new version, never copied;
* ``None`` or ``""`` — the new version has no secret (clear);
* any other string — sealed with the keyring's primary key.

KEEP is REFUSED (:class:`SecretReentryRequiredError`, API → 422) when the new
version changes ``kind`` or the ``base_url`` endpoint (scheme, host, port): the
stored key was entrusted to ONE endpoint, and carrying it to another would let a
tenant admin repoint the URL at a host they control and read the key off the
wire. Changing only the model / options / path may keep it.

Nothing here commits: the caller's session owns the transaction.
"""

from __future__ import annotations

import datetime
import enum
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit
from uuid import UUID, uuid7

from sqlalchemy import and_, func, select, text, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TenantConnection, TenantConnectionVersion
from src.services.providers.crypto import Keyring, SealedSecret


class _Keep(enum.Enum):
    KEEP = "keep"


#: Sentinel: "the request did not mention the secret — keep the stored one".
KEEP_SECRET: Final = _Keep.KEEP


class ConnectionVersionConflictError(Exception):
    """``expected_version`` does not match the live version (API → 409)."""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"expected version {expected}, current is {current}")
        self.expected = expected
        self.current = current


class SecretReentryRequiredError(Exception):
    """KEEP_SECRET across an endpoint/kind change — the secret must be re-entered (API → 422)."""

    def __init__(self) -> None:
        super().__init__(
            "re-enter the secret: the provider kind or endpoint changed, so the stored "
            "secret cannot be carried over"
        )


_DEFAULT_PORTS: Final = {"http": 80, "https": 443}


def _endpoint(base_url: str | None) -> tuple[str, str, int | None] | None:
    """``(scheme, host, port)`` a secret is sent to; ``None`` when there is no URL.

    An unparseable URL yields a value that never equals a parseable one (fail closed)."""
    if not base_url:
        return None
    try:
        parts = urlsplit(base_url.strip())
        scheme = parts.scheme.lower()
        port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)
        return scheme, (parts.hostname or "").lower(), port
    except ValueError:
        return ("<invalid>", base_url, None)


def secret_may_carry(previous: TenantConnectionVersion, *, kind: str, base_url: str | None) -> bool:
    return previous.kind == kind and _endpoint(previous.base_url) == _endpoint(base_url)


@dataclass(frozen=True, slots=True)
class VersionRef:
    """A version identified across tenants (GC / rewrap scans)."""

    tenant_id: UUID
    connection_id: UUID
    version: int


@dataclass(frozen=True, slots=True)
class StoredSecret:
    ref: VersionRef
    ciphertext: bytes
    key_id: str


async def sees_all_tenants(session: AsyncSession, *tables: str) -> bool:
    """True when ``session``'s role bypasses RLS on every one of ``tables``.

    Superuser, ``BYPASSRLS``, or (a member of) the table owner on a table that does
    not ``FORCE ROW LEVEL SECURITY``. A cross-tenant maintenance scan (GC, rewrap)
    under the restricted runtime role would see one tenant or none and report a
    vacuous success — callers must refuse instead."""
    for table in tables or ("tenant_connection_versions",):
        ok = (
            await session.execute(
                text(
                    "SELECT r.rolsuper OR r.rolbypassrls OR ("
                    "  pg_has_role(current_user, c.relowner, 'USAGE')"
                    "  AND NOT c.relforcerowsecurity) "
                    "FROM pg_class c JOIN pg_roles r ON r.rolname = current_user "
                    "WHERE c.oid = to_regclass(:t)"
                ),
                {"t": table},
            )
        ).scalar_one_or_none()
        if not ok:
            return False
    return True


class ConnectionsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── reads ────────────────────────────────────────────────────────────────

    async def _active(self, tenant_id: UUID, slot: str, *, lock: bool) -> TenantConnection | None:
        stmt = select(TenantConnection).where(
            TenantConnection.tenant_id == tenant_id,
            TenantConnection.slot == slot,
            TenantConnection.status == "active",
        )
        if lock:
            stmt = stmt.with_for_update()
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_current(
        self, tenant_id: UUID, slot: str
    ) -> tuple[TenantConnection, TenantConnectionVersion] | None:
        """The active connection of ``(tenant, slot)`` and its current version."""
        row = (
            await self.session.execute(
                select(TenantConnection, TenantConnectionVersion)
                .join(
                    TenantConnectionVersion,
                    and_(
                        TenantConnectionVersion.connection_id == TenantConnection.id,
                        TenantConnectionVersion.version == TenantConnection.current_version,
                    ),
                )
                .where(
                    TenantConnection.tenant_id == tenant_id,
                    TenantConnection.slot == slot,
                    TenantConnection.status == "active",
                )
            )
        ).first()
        return None if row is None else (row[0], row[1])

    async def get_version(
        self, tenant_id: UUID, connection_id: UUID, version: int
    ) -> TenantConnectionVersion | None:
        """A specific (pinned) version — only if its connection belongs to ``tenant_id``.
        A deleted connection's versions stay readable: a paused run may pin one."""
        return (
            await self.session.execute(
                select(TenantConnectionVersion)
                .join(
                    TenantConnection, TenantConnection.id == TenantConnectionVersion.connection_id
                )
                .where(
                    TenantConnection.tenant_id == tenant_id,
                    TenantConnectionVersion.connection_id == connection_id,
                    TenantConnectionVersion.version == version,
                )
            )
        ).scalar_one_or_none()

    async def list_for_tenant(
        self, tenant_id: UUID
    ) -> list[tuple[TenantConnection, TenantConnectionVersion]]:
        """Every ACTIVE connection of the tenant with its current version, by slot."""
        rows = (
            await self.session.execute(
                select(TenantConnection, TenantConnectionVersion)
                .join(
                    TenantConnectionVersion,
                    and_(
                        TenantConnectionVersion.connection_id == TenantConnection.id,
                        TenantConnectionVersion.version == TenantConnection.current_version,
                    ),
                )
                .where(
                    TenantConnection.tenant_id == tenant_id,
                    TenantConnection.status == "active",
                )
                .order_by(TenantConnection.slot)
            )
        ).all()
        return [(r[0], r[1]) for r in rows]

    # ── writes ───────────────────────────────────────────────────────────────

    async def create_version(
        self,
        *,
        tenant_id: UUID,
        slot: str,
        kind: str,
        base_url: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        secret: str | _Keep | None = KEEP_SECRET,
        keyring: Keyring | None = None,
        created_by: UUID | None = None,
        expected_version: int | None = None,
    ) -> tuple[TenantConnection, TenantConnectionVersion]:
        """Write the next version of ``(tenant, slot)`` (creating the connection on
        first use) and make it current; the previous version is stamped superseded.

        ``expected_version`` (0 = "there is no active connection yet") guards a
        concurrent edit: a mismatch raises :class:`ConnectionVersionConflictError`.
        A secret to seal (or carry forward) needs ``keyring``; an empty keyring
        raises ``KeyringNotConfiguredError`` before anything is written."""
        conn = await self._active(tenant_id, slot, lock=True)
        current = conn.current_version if conn is not None else 0
        if expected_version is not None and expected_version != current:
            raise ConnectionVersionConflictError(expected_version, current)

        previous: TenantConnectionVersion | None = None
        if conn is None:
            conn = TenantConnection(
                id=uuid7(),
                tenant_id=tenant_id,
                slot=slot,
                current_version=1,
                status="active",
                created_by=created_by,
                updated_by=created_by,
            )
            new_version = 1
        else:
            previous = await self.get_version(tenant_id, conn.id, conn.current_version)
            new_version = conn.current_version + 1

        if (
            secret is KEEP_SECRET
            and previous is not None
            and previous.secret_ciphertext is not None
            and not secret_may_carry(previous, kind=kind, base_url=base_url)
        ):
            raise SecretReentryRequiredError

        sealed = self._seal(
            secret,
            previous=previous,
            keyring=keyring,
            tenant_id=tenant_id,
            connection_id=conn.id,
            version=new_version,
        )

        if previous is None:
            self.session.add(conn)
            await self.session.flush()
        else:
            previous.superseded_at = datetime.datetime.now(datetime.UTC)
            conn.current_version = new_version
            conn.updated_by = created_by

        row = TenantConnectionVersion(
            connection_id=conn.id,
            version=new_version,
            kind=kind,
            base_url=base_url,
            model=model,
            options=dict(options or {}),
            secret_ciphertext=sealed.ciphertext if sealed else None,
            key_id=sealed.key_id if sealed else None,
            secret_fingerprint=sealed.fingerprint if sealed else None,
            created_by=created_by,
        )
        self.session.add(row)
        await self.session.flush()
        return conn, row

    @staticmethod
    def _seal(
        secret: str | _Keep | None,
        *,
        previous: TenantConnectionVersion | None,
        keyring: Keyring | None,
        tenant_id: UUID,
        connection_id: UUID,
        version: int,
    ) -> SealedSecret | None:
        if secret is KEEP_SECRET:
            if previous is None or previous.secret_ciphertext is None:
                return None
            ring = keyring if keyring is not None else Keyring(())
            plain = ring.decrypt(
                previous.secret_ciphertext,
                tenant_id=tenant_id,
                connection_id=previous.connection_id,
                version=previous.version,
            )
            return ring.encrypt(
                plain, tenant_id=tenant_id, connection_id=connection_id, version=version
            )
        if not secret:  # None or "" → clear
            return None
        ring = keyring if keyring is not None else Keyring(())
        return ring.encrypt(
            secret, tenant_id=tenant_id, connection_id=connection_id, version=version
        )

    async def mark_superseded(self, tenant_id: UUID, connection_id: UUID, version: int) -> bool:
        """Stamp ``superseded_at`` (idempotent: an already-stamped row keeps its time)."""
        result = await self.session.execute(
            update(TenantConnectionVersion)
            .where(
                TenantConnectionVersion.connection_id == connection_id,
                TenantConnectionVersion.version == version,
                TenantConnectionVersion.superseded_at.is_(None),
                TenantConnectionVersion.connection_id.in_(
                    select(TenantConnection.id).where(TenantConnection.tenant_id == tenant_id)
                ),
            )
            .values(superseded_at=func.now())
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    async def soft_delete(
        self,
        tenant_id: UUID,
        slot: str,
        *,
        updated_by: UUID | None,
        expected_version: int | None = None,
    ) -> bool:
        """Retire the active connection of ``(tenant, slot)`` — the slot falls back to
        the deployment default. Its current version is stamped superseded so the GC
        may purge its secret once no live run pins it. False when nothing is active.

        ``expected_version`` (the version the caller saw; 0 = none) guards a
        concurrent edit under the row lock: a mismatch raises
        :class:`ConnectionVersionConflictError` and nothing is deleted."""
        conn = await self._active(tenant_id, slot, lock=True)
        if expected_version is not None:
            current = conn.current_version if conn is not None else 0
            if current != expected_version:
                raise ConnectionVersionConflictError(expected_version, current)
        if conn is None:
            return False
        conn.status = "deleted"
        conn.deleted_at = datetime.datetime.now(datetime.UTC)
        conn.updated_by = updated_by
        await self.session.flush()
        await self.mark_superseded(tenant_id, conn.id, conn.current_version)
        return True

    async def record_test(
        self, tenant_id: UUID, connection_id: UUID, *, ok: bool | None, error: str | None = None
    ) -> bool:
        """Stamp ``last_test_*``. ``ok=None`` = "not probed" (no auth-only endpoint for the
        kind): ``last_test_ok`` stays NULL and ``error`` carries the honest explanation."""
        result = await self.session.execute(
            update(TenantConnection)
            .where(TenantConnection.id == connection_id, TenantConnection.tenant_id == tenant_id)
            .values(
                last_test_at=func.now(),
                last_test_ok=ok,
                last_test_error=None if ok else error,
            )
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    # ── secrets ──────────────────────────────────────────────────────────────

    @staticmethod
    def decrypt_secret(
        tenant_id: UUID, version: TenantConnectionVersion, keyring: Keyring
    ) -> str | None:
        """The plaintext secret of ``version`` (None when it has none / was purged).
        Fails closed — a wrong tenant, unknown kid or tampered row raises."""
        if version.secret_ciphertext is None:
            return None
        return keyring.decrypt(
            version.secret_ciphertext,
            tenant_id=tenant_id,
            connection_id=version.connection_id,
            version=version.version,
        )

    async def secrets_to_rewrap(
        self, primary_key_id: str, *, limit: int = 500
    ) -> list[StoredSecret]:
        """Stored secrets NOT sealed by ``primary_key_id`` (cross-tenant; owner session)."""
        rows = (
            await self.session.execute(
                select(
                    TenantConnection.tenant_id,
                    TenantConnectionVersion.connection_id,
                    TenantConnectionVersion.version,
                    TenantConnectionVersion.secret_ciphertext,
                    TenantConnectionVersion.key_id,
                )
                .join(
                    TenantConnection, TenantConnection.id == TenantConnectionVersion.connection_id
                )
                .where(
                    TenantConnectionVersion.secret_ciphertext.is_not(None),
                    TenantConnectionVersion.key_id != primary_key_id,
                )
                .order_by(TenantConnectionVersion.connection_id, TenantConnectionVersion.version)
                .limit(limit)
            )
        ).all()
        return [
            StoredSecret(VersionRef(r[0], r[1], r[2]), bytes(r[3]), str(r[4]))
            for r in rows
            if r[3] is not None
        ]

    async def stored_key_ids(self) -> set[str]:
        """Every kid a stored secret references (readiness: is one missing?)."""
        rows = await self.session.execute(
            select(TenantConnectionVersion.key_id)
            .where(TenantConnectionVersion.key_id.is_not(None))
            .distinct()
        )
        return {str(k) for k in rows.scalars().all() if k}

    async def replace_secret(
        self, ref: VersionRef, sealed: SealedSecret, *, from_key_id: str
    ) -> bool:
        """Swap in a re-sealed envelope — only if the row still holds ``from_key_id``
        (a concurrent purge or rewrap wins; never resurrect a purged secret)."""
        result = await self.session.execute(
            update(TenantConnectionVersion)
            .where(
                TenantConnectionVersion.connection_id == ref.connection_id,
                TenantConnectionVersion.version == ref.version,
                TenantConnectionVersion.key_id == from_key_id,
                TenantConnectionVersion.secret_ciphertext.is_not(None),
            )
            .values(
                secret_ciphertext=sealed.ciphertext,
                key_id=sealed.key_id,
                # The fingerprint is keyed by the PRIMARY key (crypto module doc) —
                # re-sealing under a new primary must recompute it, or the UI's
                # "key …ab12cd" would silently stop matching a re-entered secret.
                secret_fingerprint=sealed.fingerprint,
            )
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    # ── GC ───────────────────────────────────────────────────────────────────

    async def purge_candidates(
        self,
        *,
        superseded_before: datetime.datetime | None = None,
        exclude: Collection[tuple[UUID, int]] = (),
        limit: int = 500,
    ) -> list[VersionRef]:
        """Superseded versions still holding a secret, minus ``exclude`` — the
        ``(connection_id, version)`` refs pinned by non-terminal runs, which the GC
        computes (this query does not read ``runs``). Cross-tenant; owner session."""
        stmt = (
            select(
                TenantConnection.tenant_id,
                TenantConnectionVersion.connection_id,
                TenantConnectionVersion.version,
            )
            .join(TenantConnection, TenantConnection.id == TenantConnectionVersion.connection_id)
            .where(
                TenantConnectionVersion.superseded_at.is_not(None),
                TenantConnectionVersion.secret_ciphertext.is_not(None),
                TenantConnectionVersion.purged_at.is_(None),
            )
            .order_by(TenantConnectionVersion.superseded_at)
            .limit(limit)
        )
        if superseded_before is not None:
            stmt = stmt.where(TenantConnectionVersion.superseded_at < superseded_before)
        if exclude:
            stmt = stmt.where(
                tuple_(
                    TenantConnectionVersion.connection_id, TenantConnectionVersion.version
                ).not_in(list(exclude))
            )
        rows = (await self.session.execute(stmt)).all()
        return [VersionRef(r[0], r[1], r[2]) for r in rows]

    async def purge_secret(self, connection_id: UUID, version: int) -> bool:
        """Destroy a SUPERSEDED version's secret (the config row stays for history).
        Never touches a current version: ``superseded_at`` must be set."""
        result = await self.session.execute(
            update(TenantConnectionVersion)
            .where(
                TenantConnectionVersion.connection_id == connection_id,
                TenantConnectionVersion.version == version,
                TenantConnectionVersion.superseded_at.is_not(None),
                TenantConnectionVersion.secret_ciphertext.is_not(None),
            )
            .values(secret_ciphertext=None, key_id=None, purged_at=func.now())
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))
