"""Re-seal every stored provider secret under the keyring's PRIMARY key.

Key rotation for ``PROVIDER_SECRET_KEYS`` (``src/services/providers/crypto.py``):

1. prepend the new key: ``PROVIDER_SECRET_KEYS="new:<b64>,old:<b64>"`` and roll;  # pragma: allowlist secret
2. run this script with ``--apply`` — every secret sealed by a non-primary key is
   decrypted (with its row-bound AAD) and re-sealed by the primary;
3. once a dry run reports ``0 to rewrap``, drop the old key.

The same keyring also seals every tenant's stored LICENCE TOKEN
(``tenants.licence_token_ciphertext``, AAD ``("tenant-licence", tenant_id)`` —
migration 0089), so those are re-sealed too, by the same compare-and-swap rule.
Dropping a key before they are re-sealed would make every such tenant's licence
unreadable (the API-key lane answers 402 ``licence_unreadable``).

Each secret is re-sealed for its OWN row (tenant|connection|version), and the
update only lands if the row still holds the key it was read with — a concurrent
GC purge wins and is never resurrected. A secret that cannot be opened (its kid is
not in the keyring, or it fails authentication) is reported and left untouched,
and the run exits 2: rotation is NOT done until every secret was re-sealed.

The scan is cross-tenant, so run it on the OWNER connection (``DATABASE_URL`` —
the migration user), not the RLS-restricted runtime role, which would see only
one tenant (or none) and report a vacuous success::

    python -m scripts.rewrap_provider_secrets            # dry run
    python -m scripts.rewrap_provider_secrets --apply

The script REFUSES (exit 2) unless its role bypasses RLS on the connection
tables (superuser, BYPASSRLS, or the table owner) — see ``sees_all_tenants``.

Exit codes: 0 ok · 2 keyring empty, role cannot see every tenant, or at least one
secret could not be opened.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Tenant
from src.database.repositories.connections_repository import (
    ConnectionsRepository,
    VersionRef,
    sees_all_tenants,
)
from src.services.providers.crypto import Keyring, SecretDecryptError, tenant_licence_aad

_BATCH = 200


@dataclass
class RewrapReport:
    rewrapped: int = 0
    skipped_concurrent: int = 0
    failed: list[tuple[VersionRef, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


async def rewrap(session: AsyncSession, keyring: Keyring, *, apply: bool) -> RewrapReport:
    """Re-seal every non-primary secret visible to ``session``. Dry run (``apply``
    False) decrypts each one (so an unopenable secret is still reported) but writes
    nothing. The caller owns commit/rollback."""
    primary = keyring.primary_key_id  # raises KeyringNotConfiguredError when empty
    repo = ConnectionsRepository(session)
    report = RewrapReport()
    seen: set[tuple[object, int]] = set()
    while True:
        batch = [
            s
            for s in await repo.secrets_to_rewrap(primary, limit=_BATCH + len(seen))
            if (s.ref.connection_id, s.ref.version) not in seen
        ]
        if not batch:
            return report
        for stored in batch:
            seen.add((stored.ref.connection_id, stored.ref.version))
            try:
                sealed = keyring.rewrap(
                    stored.ciphertext,
                    tenant_id=stored.ref.tenant_id,
                    connection_id=stored.ref.connection_id,
                    version=stored.ref.version,
                )
            except SecretDecryptError as exc:
                report.failed.append((stored.ref, str(exc)))
                continue
            if sealed is None:  # pragma: no cover - the query excludes primary-sealed rows
                continue
            if not apply:
                report.rewrapped += 1
                continue
            if await repo.replace_secret(stored.ref, sealed, from_key_id=stored.key_id):
                report.rewrapped += 1
            else:
                report.skipped_concurrent += 1


@dataclass
class LicenceRewrapReport:
    rewrapped: int = 0
    skipped_concurrent: int = 0
    failed: list[tuple[UUID, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


async def rewrap_licence_tokens(
    session: AsyncSession, keyring: Keyring, *, apply: bool
) -> LicenceRewrapReport:
    """Re-seal every tenant licence token not sealed by the primary key. Dry run
    decrypts (so an unopenable one is still reported) but writes nothing. The
    update lands only if the row still holds the blob it was read with. The caller
    owns commit/rollback."""
    primary = keyring.primary_key_id  # raises KeyringNotConfiguredError when empty
    report = LicenceRewrapReport()
    rows = (
        await session.execute(
            select(Tenant.id, Tenant.licence_token_ciphertext).where(
                Tenant.licence_token_ciphertext.is_not(None),
                Tenant.licence_token_kid.is_distinct_from(primary),
            )
        )
    ).all()
    for tenant_id, ciphertext in rows:
        try:
            sealed = keyring.rewrap_with_aad(bytes(ciphertext), tenant_licence_aad(tenant_id))
        except SecretDecryptError as exc:
            report.failed.append((tenant_id, str(exc)))
            continue
        if sealed is None:  # kid column drifted from the envelope; nothing to do
            continue
        if not apply:
            report.rewrapped += 1
            continue
        new_ciphertext, kid = sealed
        result = await session.execute(
            update(Tenant)
            .where(Tenant.id == tenant_id, Tenant.licence_token_ciphertext == ciphertext)
            .values(licence_token_ciphertext=new_ciphertext, licence_token_kid=kid)
            .returning(Tenant.id)
        )
        if result.first() is not None:
            report.rewrapped += 1
        else:
            report.skipped_concurrent += 1
    return report


async def run(
    maker: Callable[[], AbstractAsyncContextManager[AsyncSession]], keyring: Keyring, *, apply: bool
) -> int:
    async with maker() as session:
        if not await sees_all_tenants(session, "tenant_connections", "tenant_connection_versions"):
            print(
                "ABORTED: this database role does not bypass row-level security on the "
                "connection tables, so it cannot see every tenant's secrets. Run with the "
                "table owner / a BYPASSRLS role (the migration DATABASE_URL).",
                file=sys.stderr,
            )
            return 2
        report = await rewrap(session, keyring, apply=apply)
        licences = await rewrap_licence_tokens(session, keyring, apply=apply)
        ok = report.ok and licences.ok
        if apply and ok:
            await session.commit()
        else:
            await session.rollback()
    verb = "rewrapped" if apply and ok else "to rewrap"
    print(f"{report.rewrapped} {verb}; {report.skipped_concurrent} changed concurrently")
    print(
        f"tenant licence tokens: {licences.rewrapped} {verb}; "
        f"{licences.skipped_concurrent} changed concurrently"
    )
    for ref, reason in report.failed:
        print(
            f"CANNOT OPEN tenant={ref.tenant_id} connection={ref.connection_id} "
            f"version={ref.version}: {reason}",
            file=sys.stderr,
        )
    for tenant_id, reason in licences.failed:
        print(f"CANNOT OPEN licence token of tenant={tenant_id}: {reason}", file=sys.stderr)
    if not ok:
        print("rotation NOT complete — nothing was written", file=sys.stderr)
        return 2
    return 0


async def _main(args: argparse.Namespace) -> int:
    from src.database.session import get_session_maker
    from src.services.providers.crypto import KeyringNotConfiguredError, get_keyring

    try:
        return await run(await get_session_maker(), get_keyring(), apply=args.apply)
    except KeyringNotConfiguredError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    raise SystemExit(asyncio.run(_main(parser.parse_args())))
