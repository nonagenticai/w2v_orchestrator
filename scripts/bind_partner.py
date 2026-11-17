"""P9 operator CLI: bind an EXISTING w2v tenant to an allow-listed partner licence.

The same operation as the super_admin route ``PUT /api/admin/tenants/{id}/partner``
— both call :func:`src.services.tenant_partner_bind.bind_tenant_to_partner`, so
every rule, refusal and audit row is the route's (allow-list, tenant exists / not
deleted, never overwrite another partner, optional licence token introspected and
stored sealed). It exists so a cutover (merqi, P9) does not need a super_admin
browser session.

* **Dry run is the DEFAULT**: every check runs — including the licence-token
  introspection, which is a read — and the transaction is rolled back. Nothing
  is written: no registry row, no binding, no audit row, no stored token.
* ``--apply`` binds and commits. Re-running is idempotent: an already-bound
  tenant reports ``bound=False`` and writes no second ``tenant.partner_bind``
  row (the same token again is a no-op too).
* The licence TOKEN is optional and is read from the environment variable NAMED
  by ``--licence-token-env`` — never from argv (``ps``/shell history), and it is
  never printed. There is no ``--licence-id``: the route retired that field
  (#1101, licence verification by introspection); a bare licence id proves
  nothing about the tenant.
* Audit: ``actor_subject_id`` is NULL (there is no human subject) and the
  ``detail`` carries ``"by": "operator-cli"`` — never a fabricated super_admin.

Run inside the orchestrator pod (it needs the pod's DB, keyring and licences env)::

    kubectl -n w2v exec deploy/w2v-orchestrator -c w2v-orchestrator -- \\
        python -m scripts.bind_partner --tenant-id T --partner-licence-id P
    # with a token: piped over stdin, so it is in no process's argv
    printf '%s\\n' "$TOKEN" | kubectl -n w2v exec -i deploy/w2v-orchestrator \\
        -c w2v-orchestrator -- sh -c 'IFS= read -r W2V_BIND_TOKEN; export W2V_BIND_TOKEN;
        exec python -m scripts.bind_partner --tenant-id T --partner-licence-id P \\
        --licence-token-env W2V_BIND_TOKEN --apply'

Exit codes: 0 ok (dry run or applied) · 1 refused by a bind rule (the route's
4xx/5xx code is printed) · 2 aborted (bad input / unexpected error), nothing written.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.repositories.admin_audit_repository import AuditRepository
from src.services.providers.crypto import Keyring
from src.services.tenant_licence import LicenceIntrospection
from src.services.tenant_partner_bind import bind_tenant_to_partner

#: The audit ``detail.by`` for every operator-CLI bind.
OPERATOR_CLI = "operator-cli"


def _redactor(secret: str | None, out: Callable[[str], None]) -> Callable[[str], None]:
    """Belt and braces: the bind never puts the token in a message, but an
    unexpected exception text could — every line is scrubbed before it leaves."""
    if not secret:
        return out
    return lambda line: out(line.replace(secret, "***"))


async def run(
    session_maker: async_sessionmaker[AsyncSession],
    licences: LicenceIntrospection,
    keyring: Keyring,
    *,
    tenant_id: UUID,
    partner_licence_id: UUID,
    token: str | None,
    apply: bool,
    out: Callable[[str], None] = print,
) -> int:
    from src.api.routes.tenancy.errors import TenancyError

    say = _redactor(token, out)
    mode = "APPLY" if apply else "DRY RUN"
    say(
        f"{mode}: bind tenant {tenant_id} -> partner licence {partner_licence_id} "
        f"(licence token: {'given' if token is not None else 'none'})"
    )
    try:
        async with session_maker() as session:
            outcome = await bind_tenant_to_partner(
                session,
                audit=AuditRepository(session),
                licences=licences,
                keyring=keyring,
                tenant_id=tenant_id,
                partner_licence_id=partner_licence_id,
                token=token,
                actor=None,
                by=OPERATOR_CLI,
                dry_run=not apply,
            )
    except TenancyError as exc:
        details = f" details={exc.details}" if exc.details else ""
        say(f"REFUSED {exc.status} {exc.code}: {exc.message}{details}")
        say(
            "Nothing written."
            if not apply
            else "The refused step committed nothing (a token-step refusal comes after the "
            "bind itself committed). Re-running is idempotent; a dry run shows the state."
        )
        return 1
    verb = "would bind" if not apply else "bound"
    say(
        f"{'BIND' if outcome.bound else 'NOOP'}     {tenant_id}: "
        + (f"{verb} to {partner_licence_id}" if outcome.bound else "already bound to this partner")
    )
    if outcome.created_registry_row:
        say(
            f"REGISTRY {tenant_id}: "
            f"{'would create' if not apply else 'created'} the tenants row (created_via=super_admin)"
        )
    if token is not None:
        if apply:
            state = "stored" if outcome.token_changed else "unchanged (same token already stored)"
        else:
            state = "verified (would be stored sealed)"
        say(f"TOKEN    {tenant_id}: {state}")
    say(f"summary: bound={outcome.bound} licence_ref={outcome.licence_ref}")
    say("DRY RUN — nothing written." if not apply else "APPLIED (committed).")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--partner-licence-id", type=UUID, required=True)
    parser.add_argument(
        "--licence-token-env",
        metavar="VAR",
        help="NAME of an env var holding the tenant's own w2v licence token (optional)",
    )
    parser.add_argument("--apply", action="store_true", help="write the bind (default: dry run)")
    return parser


def _token_from_env(var: str | None) -> str | None:
    if var is None:
        return None
    value = os.environ.get(var, "").strip()
    if not value:
        raise ValueError(f"--licence-token-env {var}: that environment variable is unset or empty")
    return value


async def _main(args: argparse.Namespace) -> int:
    token: str | None = None
    try:
        token = _token_from_env(args.licence_token_env)
        from src.api.routes._common import get_licence_keyring, get_licenses_client
        from src.database.session import get_session_maker

        return await run(
            await get_session_maker(),
            get_licenses_client(),
            get_licence_keyring(),
            tenant_id=args.tenant_id,
            partner_licence_id=args.partner_licence_id,
            token=token,
            apply=args.apply,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, exit 2, write nothing
        line = f"ABORTED, nothing written: {type(exc).__name__}: {exc}"
        _redactor(token, lambda s: print(s, file=sys.stderr))(line)
        return 2
    finally:
        # Dispose the pooled engine inside the loop, so asyncio.run's shutdown does
        # not warn about connections garbage-collected after the loop closed.
        from src.database.connection import close_engine

        await close_engine()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
